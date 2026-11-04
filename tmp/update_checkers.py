from pathlib import Path

p = Path("tmp/check_fullsup_artifacts.py")
s = p.read_text(encoding="utf-8")
s = s.replace("for r in range(21, 29)}", "for r in range(21, 33)}")
s = s.replace("    'RPCANet++': 'RPCANetpp_Seal评估产物',\n}",
              "    'RPCANet++': 'RPCANetpp_Seal评估产物',\n"
              "    'L2SKNet': 'L2SKNet_Seal评估产物',\n"
              "    'BGM': 'BGM_Seal评估产物',\n"
              "    'DRPCA-Net': 'DRPCANet_Seal评估产物',\n"
              "    'DURD': 'DURD_Seal评估产物',\n}")
p.write_text(s, encoding="utf-8")

p = Path("tmp/check_fullsup_table.py")
s = p.read_text(encoding="utf-8")
s = s.replace("for r in range(21, 29):", "for r in range(21, 33):")
s = s.replace("    'RPCANet++': 'RPCANet++_Seal实验记录.md',\n}",
              "    'RPCANet++': 'RPCANet++_Seal实验记录.md',\n"
              "    'L2SKNet': 'L2SKNet_Seal实验记录.md',\n"
              "    'BGM': 'BGM_Seal实验记录.md',\n"
              "    'DRPCA-Net': 'DRPCA-Net_Seal实验记录.md',\n"
              "    'DURD': 'DURD_Seal实验记录.md',\n}")
p.write_text(s, encoding="utf-8")
print("checkers updated for 12 rows")
