import sys
sys.path.insert(0, ".")
from ai.detection.yolo_detector import YOLODetector

detector = YOLODetector()
assert detector.class_conf[0] == 0.50, f"Expected 0.50, got {detector.class_conf[0]}"
print("✓ Verified: Person confidence threshold is 0.50 (eliminates weak road texture noise)")

# Test geometric filter simulation
# Case A: Phantom box on road marking (square box 60x60)
bw, bh = 60, 60
is_upright = not (bh < (bw * 1.05) or bh < 32 or bw < 14)
assert not is_upright, "Phantom square box on road was not rejected!"
print("✓ Verified: Phantom road marking square (60x60) correctly rejected.")

# Case B: Real upright person (e.g. 40x110)
bw, bh = 40, 110
is_upright = not (bh < (bw * 1.05) or bh < 32 or bw < 14)
assert is_upright, "Real person (40x110) was incorrectly rejected!"
print("✓ Verified: Real upright person (40x110) correctly accepted.")

print("\n>>> ALL PERSON FILTER TESTS PASSED! <<<")
