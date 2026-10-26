import CoreGraphics
import Foundation
import ImageIO

/// The S×S RGB canvas the tower sees (row stride S*3, no alpha). A frame already S×S is taken byte for byte (the
/// fixture path: the vlive1 battery canvases go in unchanged, so their verdict bits are the references'); any other
/// frame is letterboxed: scaled to fit (aspect kept, CoreGraphics high-quality interpolation), centered on black.
struct Canvas: Sendable {
    let side: Int
    var rgb: [UInt8]

    init(side: Int, rgb: [UInt8]) {
        precondition(rgb.count == side * side * 3)
        self.side = side
        self.rgb = rgb
    }

    /// A binary PPM (P6, maxval 255) of exactly side × side, byte for byte; nil otherwise.
    static func ppm(_ data: Data, side: Int = CameraConfig.canvasSide) -> Canvas? {
        guard let (w, h, body) = Self.parsePPM(data), w == side, h == side else { return nil }
        return Canvas(side: side, rgb: body)
    }

    static func parsePPM(_ data: Data) -> (Int, Int, [UInt8])? {
        let b = [UInt8](data)
        var i = 0, fields: [Int] = []
        guard b.count > 2, b[0] == 0x50, b[1] == 0x36 else { return nil }  // "P6"
        i = 2
        while fields.count < 3 && i < b.count {
            while i < b.count, b[i] == 0x20 || b[i] == 0x0A || b[i] == 0x0D || b[i] == 0x09 { i += 1 }
            if i < b.count, b[i] == 0x23 { while i < b.count, b[i] != 0x0A { i += 1 }; continue }  // # comment
            var v = 0, any = false
            while i < b.count, b[i] >= 0x30, b[i] <= 0x39 { v = v * 10 + Int(b[i] - 0x30); i += 1; any = true }
            guard any else { return nil }
            fields.append(v)
        }
        guard fields.count == 3, fields[2] == 255, i < b.count else { return nil }
        i += 1  // the single whitespace after maxval
        let (w, h) = (fields[0], fields[1])
        guard b.count - i >= w * h * 3 else { return nil }
        return (w, h, Array(b[i..<(i + w * h * 3)]))
    }

    func ppmData() -> Data {
        var d = Data("P6\n\(side) \(side)\n255\n".utf8)
        d.append(contentsOf: rgb)
        return d
    }

    /// Letterboxes RGB bytes (w × h, stride w*3) into side × side; identity (a copy) when w == h == side.
    static func letterbox(rgb: [UInt8], width w: Int, height h: Int, side: Int = CameraConfig.canvasSide) -> Canvas {
        if w == side && h == side { return Canvas(side: side, rgb: rgb) }
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        for p in 0..<(w * h) {
            rgba[p * 4] = rgb[p * 3]; rgba[p * 4 + 1] = rgb[p * 3 + 1]; rgba[p * 4 + 2] = rgb[p * 3 + 2]
        }
        let image = rgba.withUnsafeMutableBytes { raw -> CGImage? in
            CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)?.makeImage()
        }
        return image.map { letterbox($0, side: side) } ?? Canvas(side: side, rgb: [UInt8](repeating: 0, count: side * side * 3))
    }

    /// Letterboxes any CGImage (the camera path: a frame's pixel buffer wrapped as a CGImage).
    static func letterbox(_ image: CGImage, side: Int = CameraConfig.canvasSide) -> Canvas {
        var rgba = [UInt8](repeating: 0, count: side * side * 4)
        rgba.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(data: raw.baseAddress, width: side, height: side, bitsPerComponent: 8,
                                      bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
            ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))
            ctx.interpolationQuality = .high
            let s = min(Double(side) / Double(image.width), Double(side) / Double(image.height))
            let (dw, dh) = ((Double(image.width) * s).rounded(), (Double(image.height) * s).rounded())
            ctx.draw(image, in: CGRect(x: ((Double(side) - dw) / 2).rounded(.down), y: ((Double(side) - dh) / 2).rounded(.down),
                                       width: dw, height: dh))
        }
        var rgb = [UInt8](repeating: 0, count: side * side * 3)
        for p in 0..<(side * side) {
            rgb[p * 3] = rgba[p * 4]; rgb[p * 3 + 1] = rgba[p * 4 + 1]; rgb[p * 3 + 2] = rgba[p * 4 + 2]
        }
        return Canvas(side: side, rgb: rgb)
    }

    /// PHONEUX: a photo file's bytes (JPEG, HEIC, PNG…) as an upright CGImage, EXIF orientation applied, long edge at
    /// most 2048 (the canvas is 320, so nothing the engine sees is lost). nil: not an image.
    static func decode(_ data: Data, maxSide: Int = 2048) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let opts: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                     kCGImageSourceCreateThumbnailWithTransform: true,
                                     kCGImageSourceThumbnailMaxPixelSize: maxSide]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
    }

    /// The canvas as a CGImage (for the surface's frozen view and the Mac's rendered shots).
    func cgImage() -> CGImage? {
        var rgba = [UInt8](repeating: 255, count: side * side * 4)
        for p in 0..<(side * side) {
            rgba[p * 4] = rgb[p * 3]; rgba[p * 4 + 1] = rgb[p * 3 + 1]; rgba[p * 4 + 2] = rgb[p * 3 + 2]
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: side * 4,
                       space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}
