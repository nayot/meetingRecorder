// mac_tap — capture macOS system audio via a Core Audio process tap (macOS 14.2+).
//
// Usage: mac_tap <out.raw>
//
// Writes interleaved float32 PCM to <out.raw> until SIGINT/SIGTERM. Before any
// audio, prints one line to stdout: "FORMAT <sample_rate> <channels>".
// No BlackHole or Multi-Output Device needed; the tap follows whatever device
// is playing. First run triggers the "System Audio Recording" permission prompt
// for the terminal app.
//
// Compiled automatically by record.py into ~/.cache/meetingRecorder/.

import AudioToolbox
import CoreAudio
import Foundation

func fail(_ msg: String, _ status: OSStatus = 0) -> Never {
    FileHandle.standardError.write("mac_tap: \(msg)\(status != 0 ? " (OSStatus \(status))" : "")\n".data(using: .utf8)!)
    exit(1)
}

guard CommandLine.arguments.count == 2 else { fail("usage: mac_tap <out.raw>") }
let outPath = CommandLine.arguments[1]
guard FileManager.default.createFile(atPath: outPath, contents: nil),
      let out = FileHandle(forWritingAtPath: outPath) else { fail("cannot open \(outPath)") }

// 1. Global stereo tap of all processes, audio still plays normally.
let desc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
desc.isPrivate = true
desc.muteBehavior = .unmuted
desc.name = "meetingRecorder tap"
var tapID = AudioObjectID(kAudioObjectUnknown)
var err = AudioHardwareCreateProcessTap(desc, &tapID)
guard err == noErr else { fail("cannot create process tap — macOS 14.2+ and System Audio Recording permission required", err) }

// 2. Tap stream format.
var fmt = AudioStreamBasicDescription()
var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
var addr = AudioObjectPropertyAddress(
    mSelector: kAudioTapPropertyFormat,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
err = AudioObjectGetPropertyData(tapID, &addr, 0, nil, &size, &fmt)
guard err == noErr else { fail("cannot read tap format", err) }
let channels = Int(fmt.mChannelsPerFrame)
let nonInterleaved = (fmt.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0
guard (fmt.mFormatFlags & kAudioFormatFlagIsFloat) != 0, fmt.mBitsPerChannel == 32 else {
    fail("unexpected tap format (flags \(fmt.mFormatFlags), \(fmt.mBitsPerChannel) bits)")
}

// 3. A tap is read through a private aggregate device that contains it.
let aggDesc: [String: Any] = [
    kAudioAggregateDeviceNameKey: "meetingRecorder aggregate",
    kAudioAggregateDeviceUIDKey: "meetingRecorder-\(UUID().uuidString)",
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceIsStackedKey: false,
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceTapListKey: [[
        kAudioSubTapUIDKey: desc.uuid.uuidString,
        kAudioSubTapDriftCompensationKey: true,
    ]],
]
var aggID = AudioObjectID(kAudioObjectUnknown)
err = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggID)
guard err == noErr else {
    AudioHardwareDestroyProcessTap(tapID)
    fail("cannot create aggregate device", err)
}

func cleanup() {
    AudioHardwareDestroyAggregateDevice(aggID)
    AudioHardwareDestroyProcessTap(tapID)
}

// 4. IO proc: append captured frames to the file (interleaving if needed).
let queue = DispatchQueue(label: "mac_tap.io")
var procID: AudioDeviceIOProcID?
err = AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, queue) { _, inData, _, _, _ in
    let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inData))
    if abl.count == 0 { return }
    if !nonInterleaved || abl.count == 1 {
        for buf in abl where buf.mData != nil && buf.mDataByteSize > 0 {
            out.write(Data(bytes: buf.mData!, count: Int(buf.mDataByteSize)))
        }
    } else {
        let frames = Int(abl[0].mDataByteSize) / 4
        var inter = [Float](repeating: 0, count: frames * abl.count)
        for (c, buf) in abl.enumerated() {
            guard let p = buf.mData?.assumingMemoryBound(to: Float.self) else { continue }
            for f in 0..<frames { inter[f * abl.count + c] = p[f] }
        }
        inter.withUnsafeBufferPointer { out.write(Data(buffer: $0)) }
    }
}
guard err == noErr, let procID else { cleanup(); fail("cannot create IO proc", err) }

print("FORMAT \(Int(fmt.mSampleRate)) \(channels)")
fflush(stdout)

err = AudioDeviceStart(aggID, procID)
guard err == noErr else {
    AudioDeviceDestroyIOProcID(aggID, procID)
    cleanup()
    fail("cannot start capture", err)
}

// 5. Run until SIGINT/SIGTERM, then tear down cleanly.
signal(SIGINT, SIG_IGN)
signal(SIGTERM, SIG_IGN)
let stop: () -> Void = {
    AudioDeviceStop(aggID, procID)
    AudioDeviceDestroyIOProcID(aggID, procID)
    queue.sync {}
    try? out.synchronize()
    try? out.close()
    cleanup()
    exit(0)
}
let sigint = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
let sigterm = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
sigint.setEventHandler(handler: stop)
sigterm.setEventHandler(handler: stop)
sigint.resume()
sigterm.resume()
dispatchMain()
