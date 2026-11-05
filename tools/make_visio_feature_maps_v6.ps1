# 特征响应热力图 v6 的 Visio 可编辑版（v1 内容 + 统一几何 + 干净版式）（Windows + 已装 Visio，走 COM）。
# 纯版式：一行列名 + 10 行（原图 + 各模块热力图），没有标题/箭头带/逐格注记/中位数行/页脚长文。
# 用法：
#   powershell -File tools/make_visio_feature_maps_v6.ps1 `
#     -PanelRoot <...>\panels -Manifest <...>\manifest.json -OutFile <...>.vsdx
param(
    [string]$PanelRoot = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\tmp\feature_v6\panels",
    [string]$Manifest   = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\tmp\feature_v6\manifest.json",
    [string]$OutFile    = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\Ours\Feature_Map\v6\Ours_FeatureMap_v6.vsdx",
    [double]$ColW = 1.05
)

$ErrorActionPreference = 'Stop'

$data = Get-Content -Raw -Path $Manifest | ConvertFrom-Json
$labels = @($data.labels)
$samples = @($data.samples | Sort-Object order)
$files = @($samples[0].panels)   # authoritative, from the manifest

$left      = 0.30
$rowLabelW = 0.78
$gap       = 0.06
$rowGap    = 0.06
$headH     = 0.26
$legendH   = 0.50
$pageW     = $left * 2 + $rowLabelW + $labels.Count * ($ColW + $gap) - $gap
$pageH     = $left * 2 + $headH + $samples.Count * ($ColW + $rowGap) + $legendH

function Set-Cell($shp, $name, $formula) { $shp.CellsU($name).FormulaU = $formula }

function Add-Label($page, $text, $x0, $y0, $x1, $y1, $fill, $fontColor, $size = 7, $bold = 1, $align = 1) {
    $s = $page.DrawRectangle($x0, $y0, $x1, $y1)
    $s.Text = $text
    Set-Cell $s 'Char.Size'   "$size pt"
    Set-Cell $s 'Char.Color'  "RGB($($fontColor[0]),$($fontColor[1]),$($fontColor[2]))"
    Set-Cell $s 'Char.Style'  "$bold"
    try { Set-Cell $s 'Char.Font' '"Microsoft YaHei"' } catch { }
    Set-Cell $s 'FillForegnd' "RGB($($fill[0]),$($fill[1]),$($fill[2]))"
    Set-Cell $s 'LinePattern' '0'
    Set-Cell $s 'VerticalAlign' '1'
    Set-Cell $s 'Para.HorzAlign' "$align"
    return $s
}

function Add-Text($page, $text, $x0, $y0, $x1, $y1, $size = 7, $color = @(60, 60, 60), $bold = 0, $align = 0) {
    $s = $page.DrawRectangle($x0, $y0, $x1, $y1)
    $s.Text = $text
    Set-Cell $s 'Char.Size'  "$size pt"
    Set-Cell $s 'Char.Color' "RGB($($color[0]),$($color[1]),$($color[2]))"
    Set-Cell $s 'Char.Style' "$bold"
    try { Set-Cell $s 'Char.Font' '"Microsoft YaHei"' } catch { }
    Set-Cell $s 'FillPattern' '0'
    Set-Cell $s 'LinePattern' '0'
    Set-Cell $s 'VerticalAlign' '1'
    Set-Cell $s 'Para.HorzAlign' "$align"
    return $s
}

function Add-Picture($page, $path, $x0, $y0, $w, $h) {
    if (-not (Test-Path $path)) { throw "missing picture: $path" }
    $shp = $page.Import((Resolve-Path $path).Path)
    Set-Cell $shp 'Width'  "$w in"
    Set-Cell $shp 'Height' "$h in"
    Set-Cell $shp 'PinX'   "$($x0 + $w / 2) in"
    Set-Cell $shp 'PinY'   "$($y0 + $h / 2) in"
    Set-Cell $shp 'LinePattern' '0'
    return $shp
}

$headFill = @(
    @(235, 235, 235), @(235, 235, 235),
    @(222, 235, 247), @(222, 235, 247),
    @(189, 215, 238), @(189, 215, 238),
    @(237, 125, 49), @(192, 0, 0), @(127, 127, 127)
)
$headFg = @(
    @(0, 0, 0), @(0, 0, 0),
    @(0, 0, 0), @(0, 0, 0),
    @(0, 0, 0), @(0, 0, 0),
    @(255, 255, 255), @(255, 255, 255), @(255, 255, 255)
)

$visio = $null
try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    try { $visio.AlertResponse = 7 } catch { }

    $doc = $visio.Documents.Add("")
    $page = $doc.Pages.Item(1)
    $page.NameU = 'v6'
    Set-Cell $page.PageSheet 'PageWidth'  "$pageW in"
    Set-Cell $page.PageSheet 'PageHeight' "$pageH in"

    # 只有列名
    $yHeadTop = $pageH - $left
    $yHeadBottom = $yHeadTop - $headH
    for ($j = 0; $j -lt $labels.Count; $j++) {
        $x0 = $left + $rowLabelW + $j * ($ColW + $gap)
        Add-Label $page $labels[$j] $x0 $yHeadBottom ($x0 + $ColW) $yHeadTop $headFill[$j] $headFg[$j] 7 1 1 | Out-Null
    }

    # 每行：左边只写样例编号，其余九张图
    $yCursor = $yHeadBottom
    foreach ($smp in $samples) {
        $yTop = $yCursor - $rowGap
        $yImg = $yTop - $ColW
        Add-Text $page ("{0}`nF1 {1} / IoU {2}" -f $smp.stem, $smp.f1.ToString('0.000'), $smp.iou.ToString('0.000')) `
            $left ($yImg + $ColW / 2 - 0.14) ($left + $rowLabelW - $gap) ($yImg + $ColW / 2 + 0.14) 6.2 @(40, 40, 40) 1 2 | Out-Null

        $dir = Join-Path $PanelRoot (Split-Path $smp.dir -Leaf)
        for ($j = 0; $j -lt $files.Count; $j++) {
            $x0 = $left + $rowLabelW + $j * ($ColW + $gap)
            Add-Picture $page (Join-Path $dir ("roi\" + $files[$j])) $x0 $yImg $ColW $ColW | Out-Null
        }
        $yCursor = $yImg
    }

    # 底部两小段色标
    $yLegend = $left + 0.04
    $legendRoot = Split-Path $PanelRoot -Parent
    $barW = 1.45
    $barH = 0.10
    Add-Picture $page (Join-Path $legendRoot 'legend_prob.png') $left $yLegend $barW $barH | Out-Null
    Add-Text $page "feature response  low - high  (per panel 2-98 %)" ($left + $barW + 0.06) $yLegend ($left + $barW + 2.75) ($yLegend + 0.17) 5.6 @(70, 70, 70) 0 0 | Out-Null
    $xBar2 = $left + $barW + 1.75
    Add-Picture $page (Join-Path $legendRoot 'legend_feature.png') $xBar2 $yLegend $barW $barH | Out-Null
    Add-Text $page "prediction  0 - 1  (fixed)" ($xBar2 + $barW + 0.06) $yLegend ($xBar2 + $barW + 1.85) ($yLegend + 0.17) 5.6 @(70, 70, 70) 0 0 | Out-Null

    $shapeCount = $page.Shapes.Count
    $doc.SaveAs($OutFile)
    $doc.Close()
    "saved $OutFile  page ${pageW}x${pageH} in  shapes: $shapeCount"
}
finally {
    if ($visio) { try { $visio.Quit() } catch { } }
}
Get-Item $OutFile | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 2) } }
