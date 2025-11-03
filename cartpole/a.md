### Deeper dive: Step 30000
Here’s a detailed breakdown of what I saw at step 30000 across several runs:
**Run 00**
* No context 1 stabilization
**Run 01**
* Context 1: swings multiple times, fails to stabilize
* Context 10: bad first swing, second swing catches and stabilizes
* Context 20: similar to 10, but smoother
* Context 30: single swing and catch stabilize
* Context 40: same as 30, but smoother
* Context 50: looks the same as 40
**Run 02**
* Context 1: swings wildly, shoots off screen
* Context 10: stabilizes in 1 swing
* Contexts 10–50 look similar
**Run 03**
* Context 1: swings back and forth forever, no stabilization
* Context 10: failed first swing, stabilizes on second
* Context 20: stabilizes on first swing
* Contexts 20–50 identical
**Run 04**
* No context 1 stabilization
**Run 05**
* No context 1 stabilization
**Run 06**
* Context 1: swings 3 times before stabilizing, overshoots on first
* Context 10: stabilizes on first swing
* Context 20: same as 10
* Context 30: overshoots first swing, stabilizes on second
* Context 40: same as 30
* Context 50: same as 30
**Run 07**
* Context 1 stabilizes
**Run 08**
* No context 1 stabilization
**Run 09**
* No context 1 stabilization
---
So overall:
* Context 1 is inconsistent, with a mix of suboptimal stabilizations and catastrophic failures.
* Later contexts (10–50) tend to be more reliable, with smoother one- or two-swing stabilizations.
* Stabilization frequency in context 1 doesn’t improve in a straightforward way with later checkpoints.