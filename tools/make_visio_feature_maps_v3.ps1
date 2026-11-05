# 特征响应图 v3 的 Visio 可编辑版（Windows + 已装 Visio，走 COM）。
# 用法：
#   powershell -File tools/make_visio_feature_maps_v3.ps1 `
#     -PanelRoot <...>\panels -Manifest <...>\manifest.json -OutFile <...>.v3.vsdx
# 版式：一页 "v3"，顶部标题 + 两条色标 + 收敛箭头带 + 每列阶段小字 + 表头 9 列，
# 每个样例一行（左侧 F1/IoU/GT，每列图下一行小字），底部中位数行与说明。
param(
    [string]$PanelRoot = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\tmp\feature_v3\panels",
    [string]$Manifest   = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\tmp\feature_v3\manifest.json",
    [string]$OutFile    = "F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\Ours\Feature_Map\v3\Ours_FeatureMap_v3.vsdx",
    [double]$ColW = 1.00
)

$ErrorActionPreference = 'Stop'

$data = Get-Content -Raw -Path $Manifest | ConvertFrom-Json
$labels = @($data.labels)
$samples = @($data.samples | Sort-Object order)
$summary = $data.summary
$files = @(
    '01_rgb.png', '02_t.png', '03_raw_rgb.png', '04_raw_t.png',
    '05_lora_rgb.png', '06_lora_t.png', '07_bsrf.png', '08_prediction.png', '09_gt.png'
)
$stages = @(
    'input', 'input',
    'frozen SAM3', 'frozen SAM3',
    'CM-LoRA', 'CM-LoRA',
    'BSRF fused', 'final', 'GT'
)
$stageNote = @(
    '', '',
    'diffuse / blank', 'diffuse / blank',
    'one blob on target', 'one blob on target',
    'single peak', 'exact boundary', ''
)

$left      = 0.30
$rowLabelW = 1.15
$gap       = 0.05
$rowGap    = 0.07
$titleH    = 0.26
$barH      = 0.11
$bandH     = 0.22
$stageH    = 0.20
$headH     = 0.30
$capH      = 0.17
$medianH   = 0.26
$footH     = 0.82
$pageW     = $left * 2 + $rowLabelW + $labels.Count * ($ColW + $gap) - $gap
$pageH     = $left * 2 + $titleH + 0.14 + $barH + 0.10 + $bandH + $stageH + $headH `
    + $samples.Count * ($ColW + $capH + $rowGap) + $medianH + $footH

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

$kind = $null
try {
    $kind = New-Object -ComObject Visio.Application
    $kind.Visible = $false
    try { $kind.AlertResponse = 7 } catch { }

    $doc = $kind.Documents.Add("")
    $page = $doc.Pages.Item(1)
    $page.NameU = 'v3'
    Set-Cell $page.PageSheet 'PageWidth'  "$pageW in"
    Set-Cell $page.PageSheet 'PageHeight' "$pageH in"

    # 标题
    $yTitle = $pageH - $left - $titleH
    Add-Text $page ("Ours feature response (v3)   |   Seal test   |   top-$($samples.Count) by per-image F1, GT box <= $($data.max_gt_side) px   |   ROI 44 px x9   |   weights Ours/CODE/train_logs2/best.pth") `
        $left $yTitle ($pageW - $left) ($yTitle + $titleH) 9.5 @(0, 0, 0) 1 0 | Out-Null

    # 色标：概率固定 0-1；BSRF 只显示峰顶(窗口 0.65-1, gamma 2)
    $barW = 1.55
    $yBar = $yTitle - 0.14 - $barH
    $legendRoot = Split-Path $PanelRoot -Parent
    Add-Picture $page (Join-Path $legendRoot 'legend_prob.png') $left $yBar $barW $barH | Out-Null
    Add-Text $page "probability  0 - 1  (fixed)" ($left + $barW + 0.06) $yBar ($left + $barW + 1.60) ($yBar + $barH) 6.5 @(60, 60, 60) 0 0 | Out-Null
    $xBar2 = $left + $barW + 1.72
    Add-Picture $page (Join-Path $legendRoot 'legend_feature.png') $xBar2 $yBar $barW $barH | Out-Null
    Add-Text $page ("BSRF response  85 - 100 % of peak" ) ($xBar2 + $barW + 0.06) $yBar ($xBar2 + $barW + 2.35) ($yBar + $barH) 6.5 @(60, 60, 60) 0 0 | Out-Null

    # 收敛箭头带（跨 03-08）
    $yBandTop = $yBar - 0.10
    $yBandBottom = $yBandTop - $bandH
    $xB0 = $left + $rowLabelW + 2 * ($ColW + $gap)
    $xB1 = $left + $rowLabelW + 8 * ($ColW + $gap) - $gap
    $band = $page.DrawRectangle($xB0, $yBandBottom, $xB1, $yBandTop)
    $band.Text = "response converges  ->"
    Set-Cell $band 'Char.Size' '8 pt'
    Set-Cell $band 'Char.Style' '1'
    Set-Cell $band 'Char.Color' 'RGB(90,90,90)'
    Set-Cell $band 'FillForegnd' 'RGB(255,242,204)'
    Set-Cell $band 'LinePattern' '0'
    Set-Cell $band 'VerticalAlign' '1'
    Set-Cell $band 'Para.HorzAlign' '1'
    try { Set-Cell $band 'Char.Font' '"Microsoft YaHei"' } catch { }

    # 每列阶段小字 + 表头
    $yStageTop = $yBandBottom - 0.07
    $yStageBottom = $yStageTop - $stageH
    $yHeadTop = $yStageBottom
    $yHeadBottom = $yHeadTop - $headH
    for ($j = 0; $j -lt $labels.Count; $j++) {
        $x0 = $left + $rowLabelW + $j * ($ColW + $gap)
        if ($stageNote[$j] -ne '') {
            Add-Text $page ("{0}  -  {1}" -f $stages[$j], $stageNote[$j]) `
                $x0 $yStageBottom ($x0 + $ColW) $yStageTop 5.4 @(110, 110, 110) 0 1 | Out-Null
        }
        Add-Label $page $labels[$j] $x0 $yHeadBottom ($x0 + $ColW) $yHeadTop $headFill[$j] $headFg[$j] 7 1 1 | Out-Null
    }
    Add-Label $page "Sample (final F1 / IoU)" $left $yHeadBottom ($left + $rowLabelW - $gap) $yHeadTop @(242, 242, 242) @(0, 0, 0) 6.5 1 1 | Out-Null

    # 数据行
    $yCursor = $yHeadBottom
    foreach ($smp in $samples) {
        $yTop = $yCursor - $rowGap
        $yImg = $yTop - $ColW
        $yCap = $yImg - $capH
        $st = $smp.stats
        $fst = $smp.feature_stats

        Add-Label $page ("{0}`nF1 {1} / IoU {2}`nGT {3} px" -f $smp.stem,
                ($st.final.f1.ToString('0.000')), ($st.final.iou.ToString('0.000')), $fst.target_px) `
            $left $yCap ($left + $rowLabelW - $gap) $yTop @(255, 255, 255) @(0, 0, 0) 6.3 1 1 | Out-Null

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
            $color = @(80, 80, 80)
            if ($j -eq 6) { $color = @(196, 89, 17) }
            Add-Text $page $caps[$j] $x0 ($yCap + 0.01) ($x0 + $ColW) ($yCap + $capH - 0.01) 5.8 $color 0 1 | Out-Null
        }

        $yCursor = $yCap
    }

    # 中位数行
    $yMedTop = $yCursor - $rowGap
    $yMedBottom = $yMedTop - $medianH
    Add-Label $page ("median`nover $($samples.Count) samples") $left $yMedBottom ($left + $rowLabelW - $gap) $yMedTop @(242, 242, 242) @(0, 0, 0) 6.0 1 1 | Out-Null
    $med = @(
        '', '',
        ("IoU {0}" -f [math]::Round($summary.median_iou.raw_rgb, 2)),
        ("IoU {0}" -f [math]::Round($summary.median_iou.raw_t, 2)),
        ("IoU {0}" -f [math]::Round($summary.median_iou.lora_rgb, 2)),
        ("IoU {0}" -f [math]::Round($summary.median_iou.lora_t, 2)),
        ("peak {0} px / {1}x" -f [math]::Round($summary.median_peak_dist_px, 1), [math]::Round($summary.median_ratio, 2)),
        ("IoU {0}" -f [math]::Round($summary.median_iou.final, 2)),
        ''
    )
    for ($j = 0; $j -lt $med.Count; $j++) {
        if ($med[$j] -eq '') { continue }
        $x0 = $left + $rowLabelW + $j * ($ColW + $gap)
        $box = $page.DrawRectangle($x0, $yMedBottom + 0.02, $x0 + $ColW, $yMedTop - 0.02)
        $box.Text = $med[$j]
        Set-Cell $box 'Char.Size' '6.4 pt'
        Set-Cell $box 'Char.Style' '1'
        Set-Cell $box 'Char.Color' 'RGB(0,0,0)'
        Set-Cell $box 'FillForegnd' 'RGB(255,235,156)'
        Set-Cell $box 'LinePattern' '0'
        Set-Cell $box 'VerticalAlign' '1'
        Set-Cell $box 'Para.HorzAlign' '1'
        try { Set-Cell $box 'Char.Font' '"Microsoft YaHei"' } catch { }
    }

    # 底部说明（锚在页面底边，避免溢出）
    $yBottom = $left
    $yFoot2 = $yBottom + 0.02
    $yFoot1 = $yBottom + 0.34
    Add-Text $page ("ROI = square window around the GT box (side = max(44 px, bbox + 2x14 px)), upscaled so every panel is 396 px. " +
        "03-06: per-modality SAM decoder outputs (03/04 frozen SAM3; 05/06 with modality LoRA + shared LoRA3, before BSRF); the white contour is that branch's decision at p = 0.5. " +
        "07: channel-L2 of the fused FPN-72^2 feature, drawn as (v - median)/(peak - median) with only the top 85-100 % kept and a white contour at 90 % of the peak; crosshair = response peak. Dashed box = GT bbox.") `
        $left $yFoot1 ($pageW - $left) ($yFoot1 + 0.28) 5.6 @(90, 90, 90) 0 0 | Out-Null
    Add-Text $page ("08: final fused prediction, 09: GT. Every probability panel uses the same fixed 0-1 scale, so colour is comparable across columns and samples. " +
        "One cell of the fused layer covers ~9x6 native px and the 50 % level of its response hump has a ~$([math]::Round($summary.median_halo_radius_px, 0)) px radius (the resolution limit of that layer), " +
        "so 07 is a peak rather than a mask; its 90 % core is ~$([math]::Round($summary.median_core_radius_px, 1)) px in radius and the peak falls inside the GT in $($summary.peak_in_gt)/$($samples.Count) samples.") `
        $left $yFoot2 ($pageW - $left) ($yFoot2 + 0.30) 5.6 @(90, 90, 90) 0 0 | Out-Null

    $shapeCount = $page.Shapes.Count
    $doc.SaveAs($OutFile)
    $doc.Close()
    "saved $OutFile  page ${pageW}x${pageH} in  shapes on page: $shapeCount"
}
finally {
    if ($kind) { try { $kind.Quit() } catch { } }
}
Get-Item $OutFile | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 2) } }
