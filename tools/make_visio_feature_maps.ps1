# 特征响应图的 Visio 可编辑版（Windows + 已装 Visio，走 COM）。
# 用法：
#   powershell -File tools/make_visio_feature_maps.ps1 `
#     -PanelRoot <...>/panels -Manifest <...>/manifest.json -OutFile <...>.vsdx
# 版式：一页 "v1"，每行一个样例（ROI ×zoom），行首为该样例的 F1/IoU，表头为 9 列名称。
param(
    [string]$PanelRoot = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\tmp\feature_v1\panels",
    [string]$Manifest   = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\tmp\feature_v1\manifest.json",
    [string]$OutFile    = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\Ours\Feature_Map\v1\Ours_FeatureMap_v1.vsdx",
    [double]$ColW = 1.05
)

$ErrorActionPreference = 'Stop'

$data = Get-Content -Raw -Path $Manifest | ConvertFrom-Json
$labels = @($data.labels)
$samples = @($data.samples | Sort-Object order)
$files = @(
    '01_rgb.png', '02_t.png', '03_nolora_rgb.png', '04_nolora_t.png',
    '05_lora_rgb.png', '06_lora_t.png', '07_bsrf.png', '08_prediction.png', '09_gt.png'
)

$left       = 0.30
$rowLabelW  = 1.15
$gap        = 0.06
$headH      = 0.34
$rowH       = $ColW
$pageW      = $left * 2 + $rowLabelW + $labels.Count * ($ColW + $gap) - $gap
$pageH      = $left * 2 + 0.30 + $headH + $samples.Count * ($rowH + $gap) + 0.22

function Set-Cell($shp, $name, $formula) { $shp.CellsU($name).FormulaU = $formula }

function Add-Label($page, $text, $x0, $y0, $x1, $y1, $fill, $fontColor, $size = 8, $bold = 1) {
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

function Add-Picture($page, $path, $x0, $y0, $w, $h) {
    $shp = $page.Import($path)
    Set-Cell $shp 'Width'  "$w in"
    Set-Cell $shp 'Height' "$h in"
    Set-Cell $shp 'PinX'   "$($x0 + $w / 2) in"
    Set-Cell $shp 'PinY'   "$($y0 + $h / 2) in"
    Set-Cell $shp 'LinePattern' '0'
    return $shp
}

# 列配色：输入两列灰蓝，w/o CM-LoRA 浅蓝，w/ CM-LoRA 中蓝，BSRF 橙，Prediction 红，GT 灰
$headFill = @(
    @(217, 217, 217), @(217, 217, 217),
    @(189, 215, 238), @(189, 215, 238),
    @(91, 155, 213), @(91, 155, 213),
    @(237, 125, 49), @(192, 0, 0), @(127, 127, 127)
)
$headFg = @(
    @(0, 0, 0), @(0, 0, 0),
    @(0, 0, 0), @(0, 0, 0),
    @(255, 255, 255), @(255, 255, 255),
    @(255, 255, 255), @(255, 255, 255), @(255, 255, 255)
)

$visio = $null
try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    try { $visio.AlertResponse = 7 } catch { }

    $doc = $visio.Documents.Add("")
    $page = $doc.Pages.Item(1)
    $page.NameU = 'v1'
    Set-Cell $page.PageSheet 'PageWidth'  "$pageW in"
    Set-Cell $page.PageSheet 'PageHeight' "$pageH in"

    # 标题
    $title = $page.DrawRectangle($left, ($pageH - $left - 0.30), ($pageW - $left), ($pageH - $left))
    $title.Text = "Ours feature response  |  Seal test  |  top-$($samples.Count) by per-image F1  |  6x ROI  |  1105 checkpoint"
    Set-Cell $title 'Char.Size' '10 pt'
    Set-Cell $title 'Char.Style' '1'
    try { Set-Cell $title 'Char.Font' '"Microsoft YaHei"' } catch { }
    Set-Cell $title 'LinePattern' '0'
    Set-Cell $title 'FillForegnd' 'RGB(255,255,255)'

    # 表头
    $yHeadTop = $pageH - $left - 0.30 - 0.10
    $yHeadBottom = $yHeadTop - $headH
    for ($j = 0; $j -lt $labels.Count; $j++) {
        $x0 = $left + $rowLabelW + $j * ($ColW + $gap)
        Add-Label $page $labels[$j] $x0 $yHeadBottom ($x0 + $ColW) $yHeadTop $headFill[$j] $headFg[$j] 7 1 | Out-Null
    }
    $rl = Add-Label $page 'Sample (F1 / IoU)' $left $yHeadBottom ($left + $rowLabelW - $gap) $yHeadTop @(242, 242, 242) @(0, 0, 0) 7 1
    $rl | Out-Null

    # 数据行
    $yCursor = $yHeadBottom - $gap
    foreach ($smp in $samples) {
        $yRow = $yCursor - $rowH
        $stem = $smp.stem
        $tag = "$stem`nF1 $([math]::Round($smp.f1, 4))  IoU $([math]::Round($smp.iou, 4))"
        Add-Label $page $tag $left $yRow ($left + $rowLabelW - $gap) $yCursor @(255, 255, 255) @(0, 0, 0) 7 0 | Out-Null
        $dir = Join-Path $PanelRoot (Split-Path $smp.dir -Leaf)
        for ($j = 0; $j -lt $files.Count; $j++) {
            $x0 = $left + $rowLabelW + $j * ($ColW + $gap)
            $path = Join-Path $dir ("roi\" + $files[$j])
            Add-Picture $page $path $x0 $yRow $ColW $ColW | Out-Null
        }
        $yCursor = $yRow - $gap
    }

    $doc.SaveAs($OutFile)
    $doc.Close()
    "saved $OutFile  page ${pageW}x${pageH} in  shapes=$($page.Shapes.Count)"
}
finally {
    if ($visio) { try { $visio.Quit() } catch { } }
}
Get-Item $OutFile | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 2) } }
