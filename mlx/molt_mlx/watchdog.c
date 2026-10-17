/*
 * molt watchdog — an external jetsam emulator and footprint recorder.
 *
 *   watchdog <pid> <cap_mib> <interval_us> <out_csv> [ready_file]
 *
 * Polls proc_pid_rusage(pid, RUSAGE_INFO_V6) every <interval_us> microseconds,
 * appends "t_us,footprint_bytes,phase" to <out_csv>, and SIGKILLs <pid> the
 * first time the footprint exceeds <cap_mib> (cap_mib <= 0 disables the kill).
 *
 * Two footprint numbers are read on every poll:
 *   ri_phys_footprint               the instantaneous value (what jetsam checks)
 *   ri_lifetime_max_phys_footprint  the kernel ledger's running maximum
 * The second cannot miss a spike that fell between two polls, so the kill also
 * fires when *it* exceeds the cap: a short excursion over the limit is a kill,
 * as it would be on a device.  The footer records both, plus ri_energy_nj.
 *
 * Why a separate C process instead of a thread in the measured process:
 *   - a Python sampler shares the GIL with the code under test and goes quiet
 *     exactly while MLX is busy, which is when the peak happens;
 *   - SIGKILL from outside is what jetsam does; a process cannot honestly
 *     emulate being killed by deciding to exit.
 *
 * Phase labels are read from a 1-line file the worker rewrites at each phase
 * boundary (<out_csv>.phase), so every sample is attributable to a phase.
 * Exit codes: 0 target exited on its own, 9 target was killed by us,
 *             2 usage / setup error.
 */
#include <libproc.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

static void read_phase(const char *path, char *buf, size_t n) {
    FILE *f = fopen(path, "r");
    if (!f) { strncpy(buf, "-", n); return; }
    if (!fgets(buf, (int)n, f)) strncpy(buf, "-", n);
    fclose(f);
    buf[strcspn(buf, "\r\n,")] = 0;
    if (!buf[0]) strncpy(buf, "-", n);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <pid> <cap_mib> <interval_us> <out_csv> [ready_file]\n", argv[0]);
        return 2;
    }
    pid_t pid = (pid_t)atoi(argv[1]);
    double cap_mib = atof(argv[2]);
    useconds_t interval = (useconds_t)atoi(argv[3]);
    const char *out_path = argv[4];
    const char *ready = argc > 5 ? argv[5] : NULL;

    char phase_path[4096];
    snprintf(phase_path, sizeof phase_path, "%s.phase", out_path);

    FILE *out = fopen(out_path, "w");
    if (!out) { perror("fopen"); return 2; }
    setvbuf(out, NULL, _IOFBF, 1 << 20);
    fprintf(out, "t_us,footprint_bytes,phase\n");

    if (ready) { FILE *r = fopen(ready, "w"); if (r) { fputs("ready\n", r); fclose(r); } }

    uint64_t cap_bytes = cap_mib > 0 ? (uint64_t)(cap_mib * 1048576.0) : 0;
    uint64_t t0 = now_us(), peak = 0, lifetime_max = 0, energy0 = 0, energy = 0;
    uint64_t last_phase_read = 0;
    char phase[128] = "-";
    int killed = 0, have_energy0 = 0;
    const char *reason = "none";

    for (;;) {
        struct rusage_info_v6 ri;
        if (proc_pid_rusage(pid, RUSAGE_INFO_V6, (rusage_info_t *)&ri) != 0) break;  /* gone */
        if (!have_energy0) { energy0 = ri.ri_energy_nj; have_energy0 = 1; }
        energy = ri.ri_energy_nj;
        if (ri.ri_lifetime_max_phys_footprint > lifetime_max)
            lifetime_max = ri.ri_lifetime_max_phys_footprint;
        uint64_t t = now_us();
        if (t - last_phase_read > 2000) {            /* phase file every ~2 ms */
            read_phase(phase_path, phase, sizeof phase);
            last_phase_read = t;
        }
        uint64_t fp = ri.ri_phys_footprint;
        if (fp > peak) peak = fp;
        fprintf(out, "%llu,%llu,%s\n", (unsigned long long)(t - t0),
                (unsigned long long)fp, phase);
        if (cap_bytes && (fp > cap_bytes || lifetime_max > cap_bytes)) {
            kill(pid, SIGKILL);
            reason = fp > cap_bytes ? "sample" : "lifetime_max";
            fprintf(out, "%llu,%llu,KILLED\n", (unsigned long long)(now_us() - t0),
                    (unsigned long long)fp);
            killed = 1;
            break;
        }
        usleep(interval);
    }
    fprintf(out, "# peak_bytes=%llu lifetime_max_bytes=%llu killed=%d kill_reason=%s energy_nj=%llu\n",
            (unsigned long long)peak, (unsigned long long)lifetime_max, killed, reason,
            (unsigned long long)(energy - energy0));
    fclose(out);
    return killed ? 9 : 0;
}
