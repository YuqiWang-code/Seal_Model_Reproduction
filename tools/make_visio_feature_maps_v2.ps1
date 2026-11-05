# 特征响应图 v2 的 Visio 可编辑版（Windows + 已装 Visio，走 COM）。
# 用法：
#   powershell -File tools/make_visio_feature_maps_v2.ps1 `
#     -PanelRoot <...>\panels -Manifest <...>\manifest.json -OutFile <...>.vsdx
# 版式：一页 "v2"，顶部标题 + 两条色标，表头 9 列，每个样例一行（左侧 F1/IoU，
# 每列图下一行小字：分支 IoU / BSRF 峰值距离与对比度），底部说明。
param(
    [string]$PanelRoot = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\tmp\feature_v2\panels",
    [string]$Manifest   = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\tmp\feature_v2\manifest.json",
    [string]$OutFile    = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\Ours\Feature_Map\v2\Ours_FeatureMap_v2.vsdx",
    [double]$ColW = 1.00
)

$ErrorActionPreference = 'Stop'

$data = Get-Content -Raw -Path $Manifest | ConvertFrom-Json
$labels = @($data.labels)
$samples = @($data.samples | Sort-Object order)
$files = @(
    '01_rgb.png', '02_t.png', '03_raw_rgb.png', '04_raw_t.png',
    '05_lora_rgb.png', '06_lora_t.png', '07_bsrf.png', '08_prediction.png', '09_gt.png'
)

$left      = 0.28
$rowLabelW = 1.05
$gap       = 0.05
$rowGap    = 0.07
$headH     = 0.34
$capH      = 0.17
$titleH    = 0.26
$footH     = 0.26
$pageW     = $left * 2 + $rowLabelW + $labels.Count * ($ColW + $gap) - $gap
$pageH     = $left * 2 + $titleH + $headH + $samples.Count * ($ColW + $capH + $rowGap) + $footH

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
    $shp = $page.Import($path)
    Set-Cell $shp 'Width'  "$w in"
    Set-Cell $shp 'Height' "$h in"
    Set-Cell $shp 'PinX'   "$($x0 + $w / 2) in"
    Set-Cell $shp 'PinY'   "$($y0 + $h / 2) in"
    Set-Cell $shp 'LinePattern' '0'
    return $shp
}

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
    $page.NameU = 'v2'
    Set-Cell $page.PageSheet 'PageWidth'  "$pageW in"
    Set-Cell $page.PageSheet 'PageHeight' "$pageH in"

    # 标题
    $yTitle = $pageH - $left - $titleH
    Add-Text $page ("Ours feature response (v2)   |   Seal test   |   top-$($samples.Count) by per-image F1   |   ROI x6   |   final weights Ours/CODE/train_logs2/best.pth") `
        $left $yTitle ($pageW - $left) ($yTitle + $titleH) 10 @(0, 0, 0) 1 0 | Out-Null

    # 色标（概率 0-1；BSRF 逐图 2-98%）
    $barW = 1.55; $barH = 0.10
    $yBar = $yTitle - 0.16
    Add-Picture $page (Join-Path (Split-Path $PanelRoot -Parent) 'legend_prob.png') $left $yBar $barW $barH | Out-Null
    Add-Text $page "probability  0 - 1  (fixed)" ($left + $barW + 0.05) $yBar ($left + $barW + 1.55) ($yBar + $barH) 6.5 @(60, 60, 60) 0 0 | Out-Null
    $xBar2 = $left + $barW + 1.65
    Add-Picture $page (Join-Path (Split-Path $PanelRoot -Parent) 'legend_feature.png') $xBar2 $yBar $barW $barH | Out-Null
    Add-Text $page "BSRF response  low - high  (per-map 2-98%)" ($xBar2 + $barW + 0.05) $yBar ($xBar2 + $barW + 2.30) ($yBar + $barH) 6.5 @(60, 60, 60) 0 0 | Out-Null

    # 表头
    $yHeadTop = $yBar - 0.10
    $yHeadBottom = $yHeadTop - $headH
    for ($j = 0; $j -lt $labels.Count; $j++) {
        $x0 = $left + $rowLabelW + $j * ($ColW + $gap)
        Add-Label $page $labels[$j] $x0 $yHeadBottom ($x0 + $ColW) $yHeadTop $headFill[$j] $headFg[$j] 7 1 1 | Out-Null
    }
    Add-Label $page 'Sample (final F1 / IoU)' $left $yHeadBottom ($left + $rowLabelW - $gap) $yHeadTop @(242, 242, 242) @(0, 0, 0) 6.5 1 1 | Out-Null

    # 数据行
    $yCursor = $yHeadBottom
    foreach ($smp in $samples) {
        $yTop = $yCursor - $rowGap
        $yImg = $yTop - $ColW
        $yCap = $yImg - $capH
        $st = $smp.stats
        $fst = $smp.feature_stats

        Add-Label $page ("{0}`nF1 {1} / IoU {2}" -f $smp.stem,
                ([math]::Round($st.final.f1, 3)), ([math]::Round($st.final.iou, 3))) `
            $left $yCap ($left + $rowLabelW - $gap) $yTop @(255, 255, 255) @(0, 0, 0) 6.5 1 1 | Out-Null

        $dir = Join-Path $PanelRoot (Split-Path $smp.dir -Leaf)
        for ($j = 0; $j -lt $files.Count; $j++) {
            $x0 = $left + $rowLabelW + $j * ($ColW + $gap)
            Add-Picture $page (Join-Path $dir ("roi\" + $files[$j])) $x0 $yImg $ColW $ColW | Out-Null
        }

        $caps = @(
            '', '',
            ("IoU {0}" -f [math]::Round($st.raw_rgb.iou, 2)),
            ("IoU {0}" -f [math]::Round($st.raw_t.iou, 2)),
            ("IoU {0}" -f [math]::Round($st.lora_rgb.iou, 2)),
            ("IoU {0}" -f [math]::Round($st.lora_t.iou, 2)),
            ("peak {0} px / {1}x" -f [math]::Round($fst.peak_dist_px, 1), [math]::Round($fst.ratio, 2)),
            ("IoU {0}" -f [math]::Round($st.final.iou, 2)),
            ''
        )
        for ($j = 0; $j -lt $caps.Count; $j++) {
            if ($caps[$j] -eq '') { continue }
            $x0 = $left + $rowLabelW + $j * ($ColW + $gap)
            Add-Text $page $caps[$j] $x0 ($yCap + 0.01) ($x0 + $ColW) ($yCap + $capH - 0.01) 5.8 @(80, 80, 80) 0 1 | Out-Null
        }

        $yCursor = $yCap
    }

    # 底部说明
    Add-Text $page ("ROI = square window around the GT box (side = max(64 px, bbox + 2x24 px)), upscaled 6x. " +
        "03-06: per-modality SAM decoder outputs (03/04 frozen SAM3; 05/06 with modality LoRA + shared LoRA3, before BSRF). " +
        "07: channel-L2 of the fused FPN-72^2 feature, per-map 2-98% scaling (shows where the network looks, not a mask). " +
        "08: final fused prediction. 09: GT. All probability panels share the fixed 0-1 scale.") `
        $left ($left * 0.5) ($pageW - $left) ($left * 0.5 + $footH) 6 @(90, 90, 90) 0 0 | Out-Null

    $doc.SaveAs($OutFile)
    $doc.Close()
    "saved $OutFile  page ${pageW}x${pageH} in"
}
finally {
    if ($visio) { try { $visio.Quit() } catch { } }
}
Get-Item $OutFile | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 2) } }
