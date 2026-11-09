You are judging time-series forecasts. Each line of your batch file is one JSON object:
- history_last96: the last 96 observed values before the forecast starts (standardised data)
- truth: the 96 values that actually happened next
- A, B: two candidate forecasts of those 96 values

Decide which forecast is better, using the truth as reference. Judge overall closeness
to the truth, including level, direction, and the shape of cycles. Do not compute
any error metric exactly; give your honest overall judgement as a careful analyst would
by looking at the numbers. If they are genuinely indistinguishable, answer "tie".

Write one JSON line per pair to the output file: {"id": "...", "verdict": "A" | "B" | "tie", "reason": "<=15 words"}
Use only your batch file. Do not read any other file.
