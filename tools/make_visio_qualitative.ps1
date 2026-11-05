# 生成定性对比图的 Visio 可编辑版（Windows + 已装 Visio，走 COM）。
# 用法：powershell -File tools/make_visio_qualitative.ps1 -PanelRoot <...>\panels_roi -OutFile <...>.vsdx
# 每样例一页，页内 = 11 个表头矩形 + 11 张 2D 图 + 11 张 3D 图 + 1 行页脚，全部是独立形状。
param(
    [string]$PanelRoot = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\Ours\qualitative\v6_20261105\panels_diag",
    [string]$OutFile   = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\Ours\qualitative\v6_20261105\seal_qualitative_11col.vsdx"
)

$ErrorActionPreference = 'Stop'

$colW      = 1.6      # inch, one column
$left      = 0.35
$headH     = 0.42
$imgH      = 1.6      # 2D panels are square
$surfH     = 1.344    # 3D panels 960x806 -> ratio
$gap       = 0.12
$capH      = 0.28
$pageW     = $left * 2 + 11 * $colW
$pageH     = $headH + $imgH + $gap + $surfH + $gap + $capH + 2 * $left * 0.6

$colors = @{
    'Image' = @(237, 125, 49);   # ED7D31
    'Mask'  = @(248, 203, 173);  # F8CBAD
    'base'  = @(157, 195, 230);  # 9DC3E6
    'ours'  = @(46, 117, 182)    # 2E75B6
}

function Set-Cell($shp, $name, $formula) { $shp.CellsU($name).FormulaU = $formula }

function Add-Label($page, $text, $x0, $y0, $x1, $y1, $fill, $fontColor, $size = 10, $bold = 1) {
    $s = $page.DrawRectangle($x0, $y0, $x1, $y1)
    $s.Text = $text
    Set-Cell $s 'Char.Size'   "$size pt"
    Set-Cell $s 'Char.Color'  "RGB($($fontColor[0]),$($fontColor[1]),$($fontColor[2]))"
    Set-Cell $s 'Char.Style'  "$bold"
    try { Set-Cell $s 'Char.Font' '"Microsoft YaHei"' } catch { }
    Set-Cell $s 'FillForegnd' "RGB($($fill[0]),$($fill[1]),$($fill[2]))"
    Set-Cell $s 'LinePattern' '0'
    Set-Cell $s 'VerticalAlign' '1'
    return $s
}

function Get-ImgAspect($path) {
    Add-Type -AssemblyName System.Drawing
    $im = [System.Drawing.Image]::FromFile($path)
    $r = $im.Height / $im.Width
    $im.Dispose()
    return $r
}

function Add-Picture($page, $path, $x0, $y0, $w, $h) {
    $shp = $page.Import($path)
    Set-Cell $shp 'Width'  "$w in"
    Set-Cell $shp 'Height' "$h in"
    Set-Cell $shp 'PinX'   "$($x0 + $w / 2) in"
    Set-Cell $shp 'PinY'   "$($y0 + $h / 2) in"
    Set-Cell $shp 'LinePattern' '0'
    return $shp
}

$samples = Get-ChildItem -Path $PanelRoot -Directory | Sort-Object Name
"pages: $($samples.Count)  page: $pageW x $pageH in"

$visio = $null
try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    try { $visio.AlertResponse = 7 } catch { }

    $doc = $visio.Documents.Add("")
    $colNames = (Get-ChildItem (Join-Path $samples[0].FullName '2d') -Filter *.png | Sort-Object Name).BaseName

    $first = $true
    foreach ($smp in $samples) {
        if ($first) { $page = $doc.Pages.Item(1); $first = $false }
        else        { $page = $doc.Pages.Add() }
        $page.NameU = $smp.Name
        Set-Cell $page.PageSheet 'PageWidth'  "$pageW in"
        Set-Cell $page.PageSheet 'PageHeight' "$pageH in"

        $yCap = $left * 0.6
        $ySurf = $yCap + $capH + $gap
        $yImg = $ySurf + $surfH + $gap
        $yHead = $yImg + $imgH

        for ($j = 0; $j -lt $colNames.Count; $j++) {
            $file = $colNames[$j]; $name = ($file -replace '^\d+_', '') -replace '^RPCANetpp$', 'RPCANet++'
            $x0 = $left + $j * $colW

            if ($name -eq 'Image') { $fill = $colors['Image']; $fc = @(0, 0, 0) }
            elseif ($name -eq 'Mask') { $fill = $colors['Mask']; $fc = @(0, 0, 0) }
            elseif ($name -eq 'Ours') { $fill = $colors['ours']; $fc = @(255, 255, 255) }
            else { $fill = $colors['base']; $fc = @(255, 255, 255) }

            Add-Label $page $name $x0 $yHead ($x0 + $colW) ($yHead + $headH) $fill $fc 10 1 | Out-Null
            Add-Picture $page (Join-Path $smp.FullName "2d\$file.png") $x0 $yImg $colW $imgH | Out-Null
            $p3 = Join-Path $smp.FullName "3d\$file.png"; $h3 = $colW * (Get-ImgAspect $p3); Add-Picture $page $p3 $x0 $ySurf $colW $h3 | Out-Null
        }

        $cap = $page.DrawRectangle($left, $yCap, $left + 11 * $colW, $yCap + $capH)
        $cap.Text = "$($smp.Name)   |   Seal test  |   11 columns: Image / Mask / 8 baselines / Ours   |   pred >= 128"
        Set-Cell $cap 'Char.Size' '8 pt'
        Set-Cell $cap 'Char.Color' 'RGB(90,90,90)'
        try { Set-Cell $cap 'Char.Font' '"Microsoft YaHei"' } catch { }
        Set-Cell $cap 'LinePattern' '0'
        "page $($smp.Name): $($page.Shapes.Count) shapes"
    }

    $doc.SaveAs($OutFile)
    $doc.Close()
    "saved $OutFile"
}
finally {
    if ($visio) { try { $visio.Quit() } catch { } }
}
Get-Item $OutFile | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 2) } }
