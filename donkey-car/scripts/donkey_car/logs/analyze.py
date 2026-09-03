import sys, re, statistics

cte, speed = [], []
resets = 0
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if "off track" in line.lower():
        resets += 1
    m = re.match(r"cte=\s*(-?\d+\.\d+)\s+speed=\s*(-?\d+\.\d+)", line)
    if m:
        cte.append(float(m.group(1)))
        speed.append(float(m.group(2)))

if not cte:
    print("no telemetry"); sys.exit()

# skip first 5 samples (launch-from-standstill transient)
c = cte[5:]
absc = [abs(x) for x in c]
rms = (sum(x*x for x in c) / len(c)) ** 0.5
# zero crossings = oscillation proxy
zc = sum(1 for i in range(1, len(c)) if (c[i] > 0) != (c[i-1] > 0))

print(f"samples={len(c)}  resets={resets}")
print(f"mean|cte|={statistics.mean(absc):.3f}  rms={rms:.3f}  "
      f"max|cte|={max(absc):.3f}  std={statistics.pstdev(c):.3f}")
print(f"zero-crossings={zc} ({100*zc/len(c):.1f}% of samples)")
print(f"mean speed={statistics.mean(speed[5:]):.2f}")
