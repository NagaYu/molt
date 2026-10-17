# ios/ — on-device harness

What runs where, stated plainly:

| Part | Path | Built here? | Tested here? |
|---|---|---|---|
| **MoltCore** — memory probe (`phys_footprint`, kernel peak, `os_proc_available_memory`), pressure monitor, memory-ordered migration planner, Memory Budget Test, fsync telemetry | `Sources/MoltCore` | yes, `swift build` (macOS, CLT) | yes, `swift run MoltCoreSelfTest` (15 checks, macOS) |
| **MoltMLX** — mapper loader/apply and `MigrationOps` over MLX-Swift | `MoltMLX/` | **no** | **no** |
| **MoltBench app** — budget test + benchmark runner, one run per launch | `App/` | **no** | **no** |

The authoring Mac has Xcode 27.0 installed but its license is not accepted
(`sudo xcodebuild -license` is needed, which only the machine owner can run), and
no iOS device is attached. Nothing here has been run on an iPhone, and the paper
and `RESULTS.md` mark every device number as **PENDING**. The simulator is not
used for any measurement: it has no per-app memory limit.

## Why the trigger polls instead of waiting for a warning

Checked against Apple's current documentation (fetched October 2026):

* `os_proc_available_memory()` (iOS 13+): "the current memory limit minus the
  memory footprint of your app at the time of the function call"; returns 0 once
  the limit is exceeded; advisory, do not cache; the limit can change during the
  app's life cycle.
* `UIApplication.didReceiveMemoryWarningNotification` and
  `applicationDidReceiveMemoryWarning(_:)`: the app received a warning about low
  memory availability; "the system may terminate it outright" if not enough is
  released.
* `DispatchSource.makeMemoryPressureSource`: monitors the **system's** memory
  pressure condition; the source starts inactive and must be `activate()`d.
* Jetsam reports distinguish `per-process-limit` (the app crossed its own limit)
  from `vm-pageshortage` (system pressure, background apps).
* `com.apple.developer.kernel.increased-memory-limit` (iOS 15+): only some
  device models honour it.

No document promises a warning before a `per-process-limit` termination. So
`PressureMonitor.poll(needBytes:nextDemandBytes:)` reads
`os_proc_available_memory()` before **every token**, which costs one call. The
warning and the dispatch source only cause an extra check and are logged. The
`need` passed in is what the chosen schedule must allocate transiently: the
margin alone for `ordered`, destination + mapper + margin for `colocated`.

## Device protocol (≥ 3 generations, no simulator)

Suggested set, one per memory class: an A14 / 4 GB device (iPhone 12), an
A16 / 6 GB device (iPhone 14 Pro or 15), and an A18 Pro / 8 GB device
(iPhone 16 Pro). Use iOS 18 or later.

1. **Build.** `brew install xcodegen && cd ios/App && xcodegen && open MoltBench.xcodeproj`,
   set a signing team, then build to the device.
2. **Stage data.** Run `python -m bench.export_device`, then
   `xcrun devicectl device copy to --device <udid> --domain-type appDataContainer --domain-identifier dev.molt.MoltBench --source build/device --destination Documents`.
3. **Memory Budget Test (reproduce the kill first).**
   `xcrun devicectl device process launch --device <udid> dev.molt.MoltBench -- -mode budget`.
   The app allocates and touches 16 MiB per step, fsyncing a record each time, until jetsam kills it.
   Then launch with `-mode budget-report` and copy `Documents/budget-summary.json` back.
   Do this 3× with the increased-memory-limit entitlement and 3× without it.
   On the device, export the matching `JetsamEvent-*.ips` (Settings → Privacy & Security → Analytics & Improvements → Analytics Data).
   Check that `reason` is `per-process-limit` and record `rpages × pageSize`.
4. **Instruments.** Profile a `-mode bench -cond D` run and a `-cond C` run with the Allocations, VM Tracker and os_signpost instruments.
   The phase names (`snapshot`, `release_src`, `load_mapper`, `map`, `load_dst`, `transplant`) match the Mac traces.
   Export the trace files.
5. **Benchmark.** For each condition in `A A2 B C D C_ordered D_colocated` and each prompt, start a fresh launch:
   `-mode bench -pair 1.5b-0.5b -cond <c> -prompt <id> -ballastMiB <b>`.
   Set `b` from the measured limit L and the measured footprints, using the same `h` grid as `bench/run_memory.py`.
   Pull `Documents/bench-*.jsonl` back.
6. **Energy.** Use the Instruments Power Profiler on the same runs.

Copy every file back under `bench/results/device/<model-identifier>/` so the
analysis can pick it up.
