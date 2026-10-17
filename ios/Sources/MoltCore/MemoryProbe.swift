import Darwin
import Foundation
#if canImport(os)
import os
#endif

/// Reads the process's memory the way the OS accounts for it.
///
/// * `physFootprint` — `task_vm_info.phys_footprint`, the number Xcode's memory
///   gauge shows and the one the per-process jetsam limit is checked against.
/// * `peakFootprint` — `task_vm_info.ledger_phys_footprint_peak`, the kernel
///   ledger's lifetime maximum; it cannot miss a spike between two polls.
/// * `availableMemory` — on iOS, `os_proc_available_memory()`, which Apple
///   documents as "the current memory limit minus the memory footprint of your
///   app at the time of the function call", returning 0 once the limit is
///   exceeded; advisory, not to be cached.  macOS has no per-app limit, so there
///   the same arithmetic is done against an experiment-supplied cap.
public struct MemoryProbe: Sendable {
    /// Only used where `os_proc_available_memory` does not exist (macOS).
    public var emulatedCapBytes: UInt64?

    public init(emulatedCapBytes: UInt64? = nil) {
        self.emulatedCapBytes = emulatedCapBytes
    }

    public static func vmInfo() -> task_vm_info_data_t? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info : nil
    }

    public func physFootprint() -> UInt64 { Self.vmInfo()?.phys_footprint ?? 0 }

    public func peakFootprint() -> UInt64 { UInt64(max(0, Self.vmInfo()?.ledger_phys_footprint_peak ?? 0)) }

    public func residentSize() -> UInt64 { Self.vmInfo()?.resident_size ?? 0 }

    /// Bytes the app may still allocate before its limit (see type doc).
    public func availableMemory() -> UInt64 {
        #if os(iOS) || os(visionOS) || targetEnvironment(macCatalyst)
        return UInt64(os_proc_available_memory())
        #else
        guard let cap = emulatedCapBytes else { return UInt64.max }
        let fp = physFootprint()
        return cap > fp ? cap - fp : 0
        #endif
    }

    /// `true` when `availableMemory()` is the OS's own number, not an emulation.
    public static var hasNativeLimit: Bool {
        #if os(iOS) || os(visionOS) || targetEnvironment(macCatalyst)
        return true
        #else
        return false
        #endif
    }
}

public let MiB: UInt64 = 1 << 20
