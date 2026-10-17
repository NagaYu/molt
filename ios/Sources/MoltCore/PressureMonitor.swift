import Dispatch
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Every memory signal an app can observe, with the time it arrived.
public enum PressureSignal: Equatable, Sendable {
    /// `DispatchSource.makeMemoryPressureSource` — *system-wide* pressure.
    case systemWarning, systemCritical, systemNormal
    /// `UIApplication.didReceiveMemoryWarningNotification`.
    case appMemoryWarning
    /// Our own poll of `os_proc_available_memory()` fell below a threshold.
    case headroomBelow(threshold: UInt64, available: UInt64)
}

/// Combines the three signals.  Why headroom polling is the *primary* trigger
/// (documented behaviour, checked against Apple's current docs):
///
/// * The dispatch source reports changes in the *system's* memory pressure
///   condition, and it is created inactive (`activate()` must be called).
/// * The UIKit memory warning tells the app about low memory availability; the
///   delegate doc warns the system "may terminate it outright" if not enough is
///   released.  Neither is documented as firing before a `per-process-limit`
///   termination, which is what an app crossing its own limit receives.
/// * `os_proc_available_memory()` is the only API that measures distance to
///   *this app's* limit, so the decode loop polls it once per token (cheap:
///   one syscall) and the other two only make the check run sooner.
public final class PressureMonitor: @unchecked Sendable {
    public typealias Handler = (PressureSignal, UInt64) -> Void   // signal, uptime ns

    private let probe: MemoryProbe
    private let queue = DispatchQueue(label: "molt.pressure")
    private var source: DispatchSourceMemoryPressure?
    private var observer: NSObjectProtocol?
    private let handler: Handler
    public private(set) var log: [(PressureSignal, UInt64)] = []
    private let lock = NSLock()

    public init(probe: MemoryProbe, handler: @escaping Handler) {
        self.probe = probe
        self.handler = handler
    }

    private func emit(_ s: PressureSignal) {
        let t = DispatchTime.now().uptimeNanoseconds
        lock.lock(); log.append((s, t)); lock.unlock()
        handler(s, t)
    }

    public func start() {
        let src = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical],
                                                          queue: queue)
        src.setEventHandler { [weak self, weak src] in
            guard let self, let ev = src?.data else { return }
            if ev.contains(.critical) { self.emit(.systemCritical) }
            else if ev.contains(.warning) { self.emit(.systemWarning) }
            else if ev.contains(.normal) { self.emit(.systemNormal) }
        }
        src.activate()                      // created inactive, per the docs
        source = src
        #if canImport(UIKit) && !os(watchOS)
        observer = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification, object: nil,
            queue: nil) { [weak self] _ in self?.emit(.appMemoryWarning) }
        #endif
    }

    public func stop() {
        source?.cancel(); source = nil
        if let o = observer { NotificationCenter.default.removeObserver(o) }
        observer = nil
    }

    /// Called by the decode loop before every token.  Returns `true` when the
    /// remaining headroom cannot absorb `needBytes` (what the chosen migration
    /// schedule must transiently allocate) plus `nextDemandBytes`.
    @discardableResult
    public func poll(needBytes: UInt64, nextDemandBytes: UInt64 = 0) -> Bool {
        let avail = probe.availableMemory()
        let threshold = needBytes &+ nextDemandBytes
        if avail < threshold {
            emit(.headroomBelow(threshold: threshold, available: avail))
            return true
        }
        return false
    }
}
