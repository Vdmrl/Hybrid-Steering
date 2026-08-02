# Final feature composition

This is a factual experiment artifact. Judge outputs are blind and exploratory.

## Frozen configuration

| feature | rank | alpha | c |
| --- | --- | ---: | ---: |
| optimism | rank1 | 4 | 1 |
| technical | rank4 | 2 | 1 |
| numbered_list | rank4 | 4 | 1 |
| persuasive | rank1 | 8 | 1 |

RSS + clamp beta: 1.0

## Conditions

- `baseline` (baseline): joint success missing; quality delta +0.000 [+0.000, +0.000] (n=128)
- `numbered_list` (numbered_list): joint success +0.531 [+0.445, +0.617] (n=128); quality delta +0.211 [+0.096, +0.338] (n=128)
- `numbered_list+persuasive` (numbered_list+persuasive): joint success +0.094 [+0.047, +0.148] (n=128); quality delta +0.122 [+0.027, +0.219] (n=128)
- `optimism` (optimism): joint success +0.203 [+0.133, +0.273] (n=128); quality delta -0.063 [-0.147, +0.019] (n=128)
- `optimism+numbered_list` (optimism+numbered_list): joint success +0.141 [+0.086, +0.203] (n=128); quality delta -0.000 [-0.127, +0.132] (n=128)
- `optimism+numbered_list+persuasive` (optimism+numbered_list+persuasive): joint success +0.094 [+0.047, +0.148] (n=128); quality delta +0.121 [+0.026, +0.218] (n=128)
- `optimism+persuasive` (optimism+persuasive): joint success +0.227 [+0.156, +0.305] (n=128); quality delta -0.045 [-0.156, +0.054] (n=128)
- `optimism+technical` (optimism+technical): joint success +0.094 [+0.047, +0.148] (n=128); quality delta -0.012 [-0.116, +0.088] (n=128)
- `optimism+technical+numbered_list` (optimism+technical+numbered_list): joint success +0.039 [+0.008, +0.078] (n=128); quality delta +0.149 [+0.031, +0.271] (n=128)
- `optimism+technical+numbered_list+persuasive` (optimism+technical+numbered_list+persuasive): joint success +0.008 [+0.000, +0.023] (n=128); quality delta +0.161 [+0.049, +0.272] (n=128)
- `optimism+technical+persuasive` (optimism+technical+persuasive): joint success +0.055 [+0.023, +0.102] (n=128); quality delta -0.091 [-0.206, +0.021] (n=128)
- `persuasive` (persuasive): joint success +0.203 [+0.133, +0.273] (n=128); quality delta -0.027 [-0.114, +0.061] (n=128)
- `technical` (technical): joint success +0.930 [+0.883, +0.969] (n=128); quality delta +0.027 [-0.075, +0.140] (n=128)
- `technical+numbered_list` (technical+numbered_list): joint success +0.438 [+0.352, +0.523] (n=128); quality delta +0.152 [+0.046, +0.266] (n=128)
- `technical+numbered_list+persuasive` (technical+numbered_list+persuasive): joint success +0.078 [+0.039, +0.125] (n=128); quality delta +0.082 [-0.028, +0.186] (n=128)
- `technical+persuasive` (technical+persuasive): joint success +0.164 [+0.102, +0.234] (n=128); quality delta -0.082 [-0.200, +0.028] (n=128)

## Own-trait deltas against baseline

- `numbered_list`: expected-score delta +0.833 [+0.644, +1.024] (n=128)
- `optimism`: expected-score delta +0.083 [+0.030, +0.141] (n=128)
- `persuasive`: expected-score delta +0.469 [+0.337, +0.600] (n=128)
- `technical`: expected-score delta +0.587 [+0.471, +0.698] (n=128)

## Retention in full composition

- `optimism` full minus singleton: -0.083 [-0.146, -0.022] (n=128)
- `technical` full minus singleton: +0.082 [+0.005, +0.163] (n=128)
- `numbered_list` full minus singleton: -0.517 [-0.743, -0.294] (n=128)
- `persuasive` full minus singleton: -0.690 [-0.856, -0.525] (n=128)

## Leakage matrix

Entries are average expected-score changes when the row feature is added.

| added \ evaluated | optimism | technical | numbered_list | persuasive |
| --- | ---: | ---: | ---: | ---: |
| optimism | +0.074 [+0.055, +0.094] (n=1024) | -0.052 [-0.078, -0.026] (n=1024) | -0.079 [-0.133, -0.024] (n=1024) | +0.045 [+0.007, +0.083] (n=1024) |
| technical | -0.138 [-0.159, -0.117] (n=1024) | +0.635 [+0.599, +0.673] (n=1024) | -0.268 [-0.335, -0.204] (n=1024) | -0.271 [-0.315, -0.225] (n=1024) |
| numbered_list | -0.028 [-0.046, -0.010] (n=1024) | -0.110 [-0.143, -0.077] (n=1024) | +0.815 [+0.743, +0.889] (n=1024) | -0.335 [-0.381, -0.288] (n=1024) |
| persuasive | +0.073 [+0.055, +0.092] (n=1024) | +0.214 [+0.183, +0.245] (n=1024) | -0.149 [-0.212, -0.085] (n=1024) | +0.330 [+0.286, +0.375] (n=1024) |

## Judge usage

{"input_tokens": 6283035, "output_tokens": 15413, "reasoning_tokens": 0}
