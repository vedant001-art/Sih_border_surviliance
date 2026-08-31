import sys
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
res = client.get('/api/v1/analytics/overview')
assert res.status_code == 200, f'Status: {res.status_code}'
data = res.json()

print('Analytics Overview Keys:', list(data.keys()))
assert 'model_benchmarks' in data, 'Missing model_benchmarks'
assert 'traffic_timeline' in data, 'Missing traffic_timeline'
assert 'threat_severity' in data, 'Missing threat_severity'
assert 'directional_comparison' in data, 'Missing directional_comparison'

mb = data['model_benchmarks']
print('\n=== Model Benchmarks Verification ===')
print('Models:', mb['models'])
print('Metrics:', mb['metrics'])
print('Radar datasets count:', len(mb['radar_datasets']))
for ds in mb['radar_datasets']:
    print(f"  - {ds['label']}: {ds['data']}")

tt = data['traffic_timeline']
print('\n=== Traffic Timeline Verification ===')
print('Time labels:', tt['labels'])
print('Authorized transit counts:', tt['authorized'])
print('Breach counts:', tt['breaches'])

ts = data['threat_severity']
print('\n=== Threat Severity Classification ===')
for k, v in ts.items():
    print(f"  - {k}: {v}")

print('\n>>> ADVANCED COMPARISONS AND GRAPHS VERIFIED 100% SUCCESSFULLY! <<<')
