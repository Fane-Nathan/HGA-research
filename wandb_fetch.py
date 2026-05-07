import urllib.request, json, csv

API_KEY = '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'

def gql(query):
    req = urllib.request.Request(
        'https://api.wandb.ai/graphql',
        data=json.dumps({'query': query}).encode(),
        headers={
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
    )
    resp = urllib.request.urlopen(req, timeout=30)
    raw = resp.read()
    data = json.loads(raw)
    if 'errors' in data:
        print(f"  GQL errors: {data['errors']}")
    return data

# Fetch ALL runs from all projects
all_runs = []
projects = [
    ('trackmania-rl', 'trackmania-rl'),
    ('trackmania-rl', 'tmrl-test'),
    ('trackmania-rl', 'RESeL-TM'),
    ('trackmania-rl', 'tmrl-resel'),
]

for entity, project in projects:
    print(f"Fetching {entity}/{project}...")
    cursor = ""
    page = 0
    while True:
        after = f', after: "{cursor}"' if cursor else ""
        query = '''{
          project(name: "%s", entityName: "%s") {
            runs(first: 50%s) {
              edges {
                cursor
                node {
                  name
                  displayName
                  state
                  createdAt
                  heartbeatAt
                  summaryMetrics
                  tags
                }
              }
              pageInfo { hasNextPage }
            }
          }
        }''' % (project, entity, after)
        
        data = gql(query)
        if 'data' not in data or data['data'].get('project') is None:
            print(f"  No data for {entity}/{project}")
            break
        
        edges = data['data']['project']['runs']['edges']
        if not edges:
            break
        
        for edge in edges:
            node = edge['node']
            cursor = edge['cursor']
            
            summary = {}
            if node.get('summaryMetrics'):
                try:
                    summary = json.loads(node['summaryMetrics'])
                except:
                    pass
            
            # compute duration from timestamps
            try:
                from datetime import datetime
                c = datetime.fromisoformat(node['createdAt'].replace('Z','+00:00'))
                h = datetime.fromisoformat(node['heartbeatAt'].replace('Z','+00:00'))
                dur = round((h-c).total_seconds()/3600, 2)
            except:
                dur = 0
            
            all_runs.append({
                'project': project,
                'name': node.get('displayName') or node.get('name', ''),
                'state': node.get('state', ''),
                'created': node.get('createdAt', ''),
                'duration_hrs': dur,
                'tags': ','.join(node.get('tags') or []),
                'train_return': summary.get('train_return', ''),
                'test_return': summary.get('test_return', ''),
                'loss_actor': summary.get('loss_actor', ''),
                'loss_critic': summary.get('loss_critic', ''),
                'entropy_coef': summary.get('entropy_coef', ''),
                'wm_kl': summary.get('wm_kl', ''),
                'wm_total_loss': summary.get('wm_total_loss', ''),
                'debug_alpha_steer': summary.get('debug_alpha_steer', ''),
                'debug_alpha_gas': summary.get('debug_alpha_gas', ''),
                'debug_log_std_mean': summary.get('debug_log_std_mean', ''),
                'epoch': summary.get('epoch', ''),
                'memory_len': summary.get('memory_len', ''),
                'kl_div_loss': summary.get('kl_div_loss', ''),
            })
        
        has_next = data['data']['project']['runs']['pageInfo']['hasNextPage']
        page += 1
        print(f"  Page {page}: {len(edges)} runs (hasNext={has_next})")
        if not has_next:
            break

# Sort by creation date
all_runs.sort(key=lambda r: r['created'])

# Print summary table
print(f"\n{'='*140}")
print(f"TOTAL: {len(all_runs)} runs across {len(set(r['project'] for r in all_runs))} projects")
print(f"{'='*140}")
fmt = "{:>3} {:<16} {:<38} {:<8} {:>5} {:>5} {:>10} {:>10} {:>10} {:>8} {}"
print(fmt.format('#', 'Project', 'Name', 'State', 'Hrs', 'Epoch', 'TrainRet', 'TestRet', 'EntCoef', 'WM_KL', 'Created'))
print('-'*140)

for i, r in enumerate(all_runs):
    def f(v, dec=2):
        if isinstance(v, (int,float)):
            return f"{v:.{dec}f}"
        return str(v)[:10] if v != '' else '-'
    
    print(fmt.format(
        i+1,
        r['project'][:16],
        r['name'][:38],
        r['state'][:8],
        f"{r['duration_hrs']:.1f}",
        f(r['epoch'], 0),
        f(r['train_return']),
        f(r['test_return']),
        f(r['entropy_coef'], 4),
        f(r['wm_kl'], 4),
        r['created'][:10]
    ))

# Save to CSV
outpath = r'c:\Users\felix\OneDrive\Documents\tmrl-test\wandb_all_runs.csv'
with open(outpath, 'w', newline='', encoding='utf-8') as fp:
    writer = csv.DictWriter(fp, fieldnames=all_runs[0].keys())
    writer.writeheader()
    writer.writerows(all_runs)
print(f"\nSaved to {outpath}")
