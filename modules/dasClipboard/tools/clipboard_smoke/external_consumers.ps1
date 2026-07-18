param(
    [string]$Daslang = ".\bin\Release\daslang.exe"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public static class ClipboardRawReader {
    public sealed class Entry {
        public uint Format;
        public byte[] Bytes;
        public Entry(uint format, byte[] bytes) { Format = format; Bytes = bytes; }
    }

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern uint RegisterClipboardFormat(string name);
    [DllImport("user32.dll")]
    private static extern bool OpenClipboard(IntPtr owner);
    [DllImport("user32.dll")]
    private static extern bool CloseClipboard();
    [DllImport("user32.dll")]
    private static extern IntPtr GetClipboardData(uint format);
    [DllImport("user32.dll")]
    private static extern bool EmptyClipboard();
    [DllImport("user32.dll")]
    private static extern IntPtr SetClipboardData(uint format, IntPtr memory);
    [DllImport("user32.dll")]
    private static extern uint EnumClipboardFormats(uint format);
    [DllImport("kernel32.dll")]
    private static extern UIntPtr GlobalSize(IntPtr memory);
    [DllImport("kernel32.dll")]
    private static extern IntPtr GlobalAlloc(uint flags, UIntPtr bytes);
    [DllImport("kernel32.dll")]
    private static extern IntPtr GlobalFree(IntPtr memory);
    [DllImport("kernel32.dll")]
    private static extern IntPtr GlobalLock(IntPtr memory);
    [DllImport("kernel32.dll")]
    private static extern bool GlobalUnlock(IntPtr memory);

    private static void SetBytes(uint format, byte[] bytes) {
        IntPtr memory = GlobalAlloc(0x42, (UIntPtr)bytes.Length);
        if (memory == IntPtr.Zero)
            throw new InvalidOperationException("cannot allocate clipboard memory");
        IntPtr destination = GlobalLock(memory);
        if (destination == IntPtr.Zero) {
            GlobalFree(memory);
            throw new InvalidOperationException("cannot lock clipboard memory for write");
        }
        Marshal.Copy(bytes, 0, destination, bytes.Length);
        GlobalUnlock(memory);
        if (SetClipboardData(format, memory) == IntPtr.Zero) {
            GlobalFree(memory);
            throw new InvalidOperationException("cannot publish clipboard format " + format);
        }
    }

    private static byte[] ReadLockedHGlobal(uint format) {
        IntPtr memory = GetClipboardData(format);
        if (memory == IntPtr.Zero) return null;
        ulong rawSize = GlobalSize(memory).ToUInt64();
        if (rawSize == 0 || rawSize > Int32.MaxValue) return null;
        IntPtr source = GlobalLock(memory);
        if (source == IntPtr.Zero) return null;
        try {
            byte[] bytes = new byte[(int)rawSize];
            Marshal.Copy(source, bytes, 0, bytes.Length);
            return bytes;
        }
        finally { GlobalUnlock(memory); }
    }

    public static List<Entry> CaptureHGlobalFormats() {
        if (!OpenClipboard(IntPtr.Zero))
            throw new InvalidOperationException("cannot open clipboard for snapshot");
        try {
            List<Entry> result = new List<Entry>();
            uint format = 0;
            while ((format = EnumClipboardFormats(format)) != 0) {
                byte[] bytes = ReadLockedHGlobal(format);
                if (bytes != null) result.Add(new Entry(format, bytes));
            }
            return result;
        }
        finally { CloseClipboard(); }
    }

    public static void RestoreHGlobalFormats(List<Entry> entries) {
        if (!OpenClipboard(IntPtr.Zero))
            throw new InvalidOperationException("cannot open clipboard for restore");
        try {
            if (!EmptyClipboard())
                throw new InvalidOperationException("cannot empty clipboard for restore");
            foreach (Entry entry in entries)
                SetBytes(entry.Format, entry.Bytes);
        }
        finally { CloseClipboard(); }
    }

    public static void SetAll(string text, byte[] html, byte[] markdown,
                              byte[] uriList, byte[] dibV5) {
        uint htmlFormat = RegisterClipboardFormat("HTML Format");
        uint markdownFormat = RegisterClipboardFormat("text/markdown");
        uint uriFormat = RegisterClipboardFormat("text/uri-list");
        if (htmlFormat == 0 || markdownFormat == 0 || uriFormat == 0 ||
            !OpenClipboard(IntPtr.Zero))
            throw new InvalidOperationException("cannot open clipboard for write");
        try {
            if (!EmptyClipboard())
                throw new InvalidOperationException("cannot empty clipboard");
            SetBytes(13, System.Text.Encoding.Unicode.GetBytes(text + "\0"));
            SetBytes(htmlFormat, AppendZero(html));
            SetBytes(markdownFormat, AppendZero(markdown));
            SetBytes(uriFormat, AppendZero(uriList));
            SetBytes(17, dibV5);
        }
        finally { CloseClipboard(); }
    }

    private static byte[] AppendZero(byte[] source) {
        byte[] result = new byte[source.Length + 1];
        Buffer.BlockCopy(source, 0, result, 0, source.Length);
        return result;
    }

    public static byte[] Get(string formatName) {
        uint format = RegisterClipboardFormat(formatName);
        if (format == 0 || !OpenClipboard(IntPtr.Zero))
            throw new InvalidOperationException("cannot open clipboard");
        try {
            byte[] bytes = ReadLockedHGlobal(format);
            if (bytes == null)
                throw new InvalidOperationException("clipboard format is absent: " + formatName);
            int size = bytes.Length;
            while (size > 0 && bytes[size - 1] == 0) --size;
            if (size == bytes.Length) return bytes;
            byte[] trimmed = new byte[size];
            Buffer.BlockCopy(bytes, 0, trimmed, 0, size);
            return trimmed;
        }
        finally { CloseClipboard(); }
    }
}
"@

$arrow = [char]0x2192
$dash = [char]0x2014
$document = "$([char]0x6587)$([char]0x66F8)"
$hello = "$([char]0x041F)$([char]0x0440)$([char]0x0438)$([char]0x0432)$([char]0x0435)$([char]0x0442)"
$naive = "na$([char]0x00EF)ve"
$dasMarker = "daScript $arrow .NET clipboard smoke $dash $document"
$externalMarker = ".NET $arrow daScript clipboard smoke $dash $hello"
$dasMarkdown = "**daScript to .NET** $dash [$document](editor://workspace/$document)"
$dasUriList = "editor://workspace/$document`r`nhttps://example.com/$naive`r`n"
$externalMarkdown = "**.NET to daScript** $dash $hello"
$externalHtmlFragment = "<p><strong>.NET to daScript</strong> $dash $hello</p>"
$probe = Join-Path $PSScriptRoot "external_probe.das"
$utf8 = [System.Text.UTF8Encoding]::new($false)

function Get-ClipboardBytes([object]$Value) {
    if ($Value -is [byte[]]) { return [byte[]]$Value }
    if ($Value -is [System.IO.MemoryStream]) { return $Value.ToArray() }
    if ($Value -is [System.IO.Stream]) {
        $copy = [System.IO.MemoryStream]::new()
        $oldPosition = $Value.Position
        $Value.Position = 0
        $Value.CopyTo($copy)
        $Value.Position = $oldPosition
        return $copy.ToArray()
    }
    if ($Value -is [string]) { return $utf8.GetBytes([string]$Value) }
    throw "unsupported clipboard object type: $($Value.GetType().FullName)"
}

function Get-ClipboardUtf8([object]$Value) {
    $bytes = Get-ClipboardBytes $Value
    $size = $bytes.Length
    while ($size -gt 0 -and $bytes[$size - 1] -eq 0) { $size-- }
    return $utf8.GetString($bytes, 0, $size)
}

function New-CfHtml([string]$Fragment) {
    $prefix = "<html><body>`r`n<!--StartFragment-->"
    $suffix = "<!--EndFragment-->`r`n</body></html>"
    $template = "Version:0.9`r`nStartHTML:{0:D10}`r`nEndHTML:{1:D10}`r`nStartFragment:{2:D10}`r`nEndFragment:{3:D10}`r`n"
    $zeroHeader = $template -f 0, 0, 0, 0
    $startHtml = $utf8.GetByteCount($zeroHeader)
    $startFragment = $startHtml + $utf8.GetByteCount($prefix)
    $endFragment = $startFragment + $utf8.GetByteCount($Fragment)
    $endHtml = $endFragment + $utf8.GetByteCount($suffix)
    return $template -f $startHtml, $endHtml, $startFragment, $endFragment `
        | ForEach-Object { $_ + $prefix + $Fragment + $suffix }
}

function Set-Int32([byte[]]$Bytes, [int]$Offset, [int]$Value) {
    [Array]::Copy([BitConverter]::GetBytes($Value), 0, $Bytes, $Offset, 4)
}

function Set-UInt32([byte[]]$Bytes, [int]$Offset, [uint32]$Value) {
    [Array]::Copy([BitConverter]::GetBytes($Value), 0, $Bytes, $Offset, 4)
}

function Set-UInt16([byte[]]$Bytes, [int]$Offset, [uint16]$Value) {
    [Array]::Copy([BitConverter]::GetBytes($Value), 0, $Bytes, $Offset, 2)
}

function New-DibV5Probe {
    [byte[]]$dib = [byte[]]::new(124 + 16)
    Set-Int32 $dib 0 124
    Set-Int32 $dib 4 2
    Set-Int32 $dib 8 -2
    Set-UInt16 $dib 12 1
    Set-UInt16 $dib 14 32
    Set-UInt32 $dib 16 3
    Set-UInt32 $dib 20 16
    Set-UInt32 $dib 40 0x00ff0000
    Set-UInt32 $dib 44 0x0000ff00
    Set-UInt32 $dib 48 0x000000ff
    Set-UInt32 $dib 52 ([uint32]::Parse("4278190080"))
    [byte[]]$pixels = @(
        0, 0, 255, 255, 0, 255, 0, 255,
        255, 0, 0, 255, 255, 255, 255, 255
    )
    [Array]::Copy($pixels, 0, $dib, 124, $pixels.Length)
    return $dib
}

function Invoke-Probe([string]$Mode) {
    & $Daslang $probe -- $Mode
    if ($LASTEXITCODE -ne 0) { throw "daScript clipboard probe '$Mode' failed ($LASTEXITCODE)" }
}

$original = [ClipboardRawReader]::CaptureHGlobalFormats()
$ownsClipboard = $false

try {
    $ownsClipboard = $true
    Invoke-Probe "write"

    $offered = [System.Windows.Forms.Clipboard]::GetDataObject()
    if ([System.Windows.Forms.Clipboard]::GetText() -ne $dasMarker) {
        throw "WinForms did not consume daScript plain UTF-8 text"
    }
    $html = Get-ClipboardUtf8 ($offered.GetData([System.Windows.Forms.DataFormats]::Html, $false))
    if (!$html.StartsWith("Version:0.9") -or
        !$html.Contains("<strong>daScript to .NET</strong>")) {
        $htmlObject = $offered.GetData([System.Windows.Forms.DataFormats]::Html, $false)
        $prefixSize = [Math]::Min(48, $html.Length)
        $prefix = $html.Substring(0, $prefixSize).Replace("`r", "\r").Replace("`n", "\n")
        $hasStrong = $html.Contains("<strong>daScript to .NET</strong>")
        $hasDocument = $html.Contains($document)
        throw "WinForms did not consume daScript CF_HTML (type=$($htmlObject.GetType().FullName), length=$($html.Length), strong=$hasStrong, unicode=$hasDocument, prefix='$prefix')"
    }
    $rawHtml = $utf8.GetString([ClipboardRawReader]::Get("HTML Format"))
    if (!$rawHtml.StartsWith("Version:0.9") -or
        !$rawHtml.Contains("<strong>daScript to .NET</strong>") -or
        !$rawHtml.Contains($document)) {
        throw "raw Win32 consumer did not receive UTF-8 CF_HTML"
    }
    $markdown = Get-ClipboardUtf8 ($offered.GetData("text/markdown", $false))
    if ($markdown -ne $dasMarkdown) {
        throw "WinForms did not consume daScript Markdown bytes"
    }
    $uriList = Get-ClipboardUtf8 ($offered.GetData("text/uri-list", $false))
    if ($uriList -ne $dasUriList) {
        throw "WinForms did not consume daScript URI-list bytes"
    }
    $image = [System.Windows.Forms.Clipboard]::GetImage()
    if ($null -eq $image -or $image.Width -ne 2 -or $image.Height -ne 2 -or
        $image.GetPixel(0, 0).ToArgb() -ne [System.Drawing.Color]::Red.ToArgb() -or
        $image.GetPixel(1, 0).ToArgb() -ne [System.Drawing.Color]::Lime.ToArgb() -or
        $image.GetPixel(0, 1).ToArgb() -ne [System.Drawing.Color]::Blue.ToArgb() -or
        $image.GetPixel(1, 1).ToArgb() -ne [System.Drawing.Color]::White.ToArgb()) {
        throw "WinForms did not consume daScript RGBA8 image"
    }
    $image.Dispose()

    [ClipboardRawReader]::SetAll(
        $externalMarker,
        $utf8.GetBytes((New-CfHtml $externalHtmlFragment)),
        $utf8.GetBytes($externalMarkdown),
        $utf8.GetBytes("https://example.com/from-dotnet`r`n"),
        (New-DibV5Probe))

    Invoke-Probe "read"
    Write-Host "PASS: daScript <-> WinForms plain/HTML/Markdown/URI/image interoperability"
}
finally {
    $currentText = [System.Windows.Forms.Clipboard]::GetText()
    if ($ownsClipboard -and ($currentText -eq $dasMarker -or $currentText -eq $externalMarker)) {
        [ClipboardRawReader]::RestoreHGlobalFormats($original)
    } elseif ($ownsClipboard) {
        Write-Warning "Clipboard changed during smoke; preserving the newer content instead of restoring."
    }
}
