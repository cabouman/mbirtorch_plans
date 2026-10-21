#!/usr/bin/env python3
"""Read the GPU queues of the group's clusters and write a web page that compares them.

For every machine in machines.json the script logs in by ssh, and for every queue it reads:
the maximum job length, the GPUs free and total, the jobs waiting and running, Slurm's own
estimate of when a typical job would start (sbatch --test-only, nothing is submitted), and
the actual waits of your own jobs in that queue over the last 30 days.

A machine that cannot be reached keeps its last numbers, marked stale with the login that
would refresh them.  Results go to queues.json and queues.html in the output directory,
which is "out_dir" in ~/.config/gpu-queues/config.json (default ~/.cache/gpu-queues).

Usage: python3 collect.py [machine ...]     (default: every machine)
"""
import datetime as dt
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.expanduser('~/.config/gpu-queues/config.json')


def load_config():
    """Return the per-laptop settings; a missing or unreadable file gives the defaults."""
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


OUT_DIR = os.path.expanduser(load_config().get('out_dir') or '~/.cache/gpu-queues')
JSON_PATH = os.path.join(OUT_DIR, 'queues.json')
HTML_PATH = os.path.join(OUT_DIR, 'queues.html')
SSH_TIMEOUT_S = 120

# Node states (sinfo StateCompact, matched as substrings) whose GPUs do not count as free.
BAD_STATES = ('drain', 'drng', 'down', 'resv', 'maint', 'comp', 'fail', 'unk', 'inval', 'plnd',
              'boot', 'pow', 'futr')
MAINT_HORIZON_DAYS = 7   # report maintenance that is active or starts within this many days
MAINT_MIN_COVERAGE = 0.1  # ignore maintenance on less than this fraction of a queue's nodes
MAINT_BLOCKS = 0.5       # a queue drops out of "best bet" when maintenance covers this much of it
MAINT_MAX_DAYS = 30      # a longer MAINT reservation is a hold on a few broken or test nodes, not a window
MAINT_SOON_MINUTES = 60  # nodes whose maintenance starts sooner than this do not count as free


def remote_script(machine):
    """Return the bash script that prints one machine's queue data as tagged lines."""
    lines = ['set +e', 'export LC_ALL=C', 'echo "NOW|$(date +%Y-%m-%dT%H:%M:%S)"']
    if machine.get('account') == 'auto':
        lines.append("ACC=$(sacctmgr -n show assoc user=$USER format=Account%40 | tr -d ' ' "
                     "| grep -i gpu | head -1)")
        lines.append('echo "ACCOUNT|$ACC"')
    for q in machine['queues']:
        p, qos, test = q['partition'], q.get('qos', ''), q['test']
        job_qos = q.get('job_qos', qos)
        lines.append(f'echo "QUEUE|{q["name"]}"')
        lines.append(f'echo "MAXP|$(scontrol show partition {p} 2>/dev/null | grep -o "MaxTime=[^ ]*")"')
        if qos:
            lines.append(f'echo "MAXQ|$(sacctmgr -n show qos {qos} format=MaxWall 2>/dev/null | tr -d " ")"')
        lines.append(f'sinfo -p {p} -h -N -O "NodeHost:24,Gres:60,GresUsed:80,StateCompact:14" 2>/dev/null '
                     f'| sed "s/^/NODE|/"')
        qflag = f'-q {job_qos}' if job_qos else ''
        lines.append(f'echo "PD|$(squeue -p {p} {qflag} -h -t PD 2>/dev/null | wc -l)"')
        lines.append(f'echo "R|$(squeue -p {p} {qflag} -h -t R 2>/dev/null | wc -l)"')
        lines.append(f'echo "EST|$(sbatch --test-only {test} --wrap hostname 2>&1 | tr "\\n" " ")"')
        lines.append('echo "END|"')
    lines.append('sacct -u $USER -S now-30days -X -P -n -o Partition,QOS,State,Submit,Start 2>/dev/null '
                 '| sed "s/^/ACCT|/"')
    # Maintenance reservations, each followed by its expanded node list.
    lines.append('scontrol show res -o 2>/dev/null | grep "Flags=[^ ]*MAINT" | while read -r line; do '
                 'echo "RES|$line"; '
                 'nl=$(printf "%s" "$line" | grep -o " Nodes=[^ ]*" | cut -d= -f2); '
                 'echo "RESNODES|$(scontrol show hostnames "$nl" 2>/dev/null | tr "\\n" " ")"; done')
    return '\n'.join(lines) + '\n'


def run_ssh(host, script):
    """Run the script on the host; return (ok, text)."""
    cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'LogLevel=ERROR', host, 'bash -s']
    try:
        r = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=SSH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, 'ssh timed out'
    if r.returncode == 255 or 'QUEUE|' not in r.stdout:
        return False, (r.stderr or r.stdout).strip()[-300:]
    return True, r.stdout


def parse_gpu_count(text):
    """Return the GPU count in a Gres string such as gpu:4, gpu:h100:8(IDX:0-7) or gpu:a100:4(S:0-1)."""
    m = re.search(r'gpu(?::[A-Za-z0-9_\-]+)?:(\d+)', text)
    return int(m.group(1)) if m else 0


def parse_time(s):
    """Return minutes for a Slurm time such as 4:00:00, 14-00:00:00 or 30:00; None if unlimited or empty."""
    s = s.strip()
    if not s or s.upper() in ('UNLIMITED', 'INFINITE', 'N/A'):
        return None
    days = 0
    if '-' in s:
        d, s = s.split('-', 1)
        days = int(d)
    parts = [int(x) for x in s.split(':')]
    if len(parts) == 3:
        h, m, _ = parts
    elif len(parts) == 2:
        h, m = 0, parts[0]
    else:
        h, m = 0, parts[0]
    return days * 24 * 60 + h * 60 + m


def fmt_minutes(mins):
    if mins is None:
        return 'no limit'
    if mins < 60:
        return f'{mins} min'
    if mins < 48 * 60:
        return f'{mins / 60:.1f} h'
    return f'{mins / 1440:.1f} days'


def parse_slurm_date(s):
    try:
        return dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        return None


def to_local(t, remote_now, now):
    """Convert a cluster-clock time to the laptop's clock, using the two clocks' offset rounded to 15 min."""
    offset = round((now - remote_now).total_seconds() / 900) * 900
    return t + dt.timedelta(seconds=offset)


def fmt_when(t, now):
    """Format a time as 'Wed Oct 21 09:00', adding the year when it is not this year."""
    return t.strftime('%a %b %d %H:%M' if t.year == now.year else '%a %b %d %Y %H:%M')


def is_window(r):
    """True for a maintenance window: a MAINT reservation with known times lasting at most MAINT_MAX_DAYS."""
    return bool(r['start'] and r['end']) and (r['end'] - r['start']).total_seconds() <= MAINT_MAX_DAYS * 86400


def maintenance_for(hosts, reservations, remote_now, now):
    """Return the first maintenance window that matters for a queue with these node names, or None.

    A window matters when it has not ended, starts within MAINT_HORIZON_DAYS, and covers at least
    MAINT_MIN_COVERAGE of the queue's nodes.  Times in the result are on the laptop's clock."""
    if not hosts:
        return None
    found = []
    for r in reservations:
        start, end = r['start'], r['end']
        if not is_window(r) or end <= remote_now:
            continue
        if (start - remote_now).total_seconds() > MAINT_HORIZON_DAYS * 86400:
            continue
        coverage = len(hosts & r['hosts']) / len(hosts)
        if coverage < MAINT_MIN_COVERAGE:
            continue
        found.append((start, {
            'name': r['name'], 'coverage': round(coverage, 3), 'active': start <= remote_now,
            'starts_in_minutes': max(int((start - remote_now).total_seconds() // 60), 0),
            'start': fmt_when(to_local(start, remote_now, now), now),
            'end': fmt_when(to_local(end, remote_now, now), now),
        }))
    return min(found, key=lambda x: x[0])[1] if found else None


def parse_output(machine, text, now):
    """Turn the tagged lines into per-queue records."""
    queues, acct_rows, account, reservations = [], [], '', []
    cur = None
    remote_now = now
    for line in text.splitlines():
        if '|' not in line:
            continue
        tag, val = line.split('|', 1)
        if tag == 'NOW':
            # Estimates are measured against the cluster's own clock, which may be in another time zone.
            remote_now = parse_slurm_date(val.strip()) or now
        elif tag == 'RES':
            fields = dict(re.findall(r'(\w+)=(\S*)', val))
            reservations.append({'name': fields.get('ReservationName', ''),
                                 'start': parse_slurm_date(fields.get('StartTime', '')),
                                 'end': parse_slurm_date(fields.get('EndTime', '')), 'hosts': set()})
        elif tag == 'RESNODES':
            if reservations:
                reservations[-1]['hosts'] = set(val.split())
        elif tag == 'ACCOUNT':
            account = val.strip()
        elif tag == 'QUEUE':
            cur = {'name': val, 'nodes': [], 'maxp': '', 'maxq': '', 'pd': 0, 'r': 0, 'est': ''}
            queues.append(cur)
        elif cur is None:
            if tag == 'ACCT':
                acct_rows.append(val.split('|'))
            continue
        elif tag == 'MAXP':
            cur['maxp'] = val.replace('MaxTime=', '')
        elif tag == 'MAXQ':
            cur['maxq'] = val
        elif tag == 'NODE':
            cur['nodes'].append(val)
        elif tag == 'PD':
            cur['pd'] = int(val.strip() or 0)
        elif tag == 'R':
            cur['r'] = int(val.strip() or 0)
        elif tag == 'EST':
            cur['est'] = val.strip()
        elif tag == 'END':
            cur = None
        elif tag == 'ACCT':
            acct_rows.append(val.split('|'))

    # Nodes about to go into maintenance can take only jobs that end first, so they do not count as free.
    soon = set()
    for r in reservations:
        if r['start'] and 0 <= (r['start'] - remote_now).total_seconds() <= MAINT_SOON_MINUTES * 60:
            soon |= r['hosts']

    spec_by_name = {q['name']: q for q in machine['queues']}
    records = []
    for q in queues:
        spec = spec_by_name.get(q['name'], {})
        total = free = 0
        hosts = set()
        for n in q['nodes']:
            parts = n.split()
            if len(parts) < 4:
                continue
            host, gres, used, state = parts[0], parts[1], parts[2], parts[3]
            hosts.add(host)
            t = parse_gpu_count(gres)
            u = parse_gpu_count(used)
            total += t
            if host not in soon and not any(b in state.lower() for b in BAD_STATES):
                free += max(t - u, 0)
        limits = [x for x in (parse_time(q['maxp']), parse_time(q['maxq'])) if x is not None]
        max_minutes = min(limits) if limits else None

        est_text, est_minutes = '', None
        m = re.search(r'to start at (\S+)', q['est'])
        if m:
            t0 = parse_slurm_date(m.group(1))
            if t0:
                est_minutes = max(int((t0 - remote_now).total_seconds() // 60), 0)
                est_text = fmt_minutes(est_minutes) if est_minutes else 'now'
        elif 'error' in q['est'].lower() or 'failure' in q['est'].lower():
            est_text = 'test job refused: ' + re.sub(r'sbatch: error: ', '', q['est'])[:160]

        waits = []
        for row in acct_rows:
            if len(row) < 5:
                continue
            part, qos, state, submit, start = row[:5]
            if part != spec.get('partition'):
                continue
            job_qos = spec.get('job_qos', spec.get('qos', ''))
            if job_qos and qos != job_qos:
                continue
            s0, s1 = parse_slurm_date(submit), parse_slurm_date(start)
            if s0 and s1 and s1 >= s0:
                waits.append(int((s1 - s0).total_seconds() // 60))
        recent = ''
        if waits:
            recent = (f'{len(waits)} jobs: median {fmt_minutes(int(statistics.median(waits)))}, '
                      f'longest {fmt_minutes(max(waits))}')

        records.append({
            'name': q['name'], 'gpu': spec.get('gpu', ''), 'gpus_free': free, 'gpus_total': total,
            'waiting': q['pd'], 'running': q['r'], 'max_job': fmt_minutes(max_minutes),
            'max_minutes': max_minutes, 'slurm_estimate': est_text, 'slurm_estimate_minutes': est_minutes,
            'recent_waits': recent, 'recent_wait_minutes': waits,
            'maintenance': maintenance_for(hosts, reservations, remote_now, now),
        })
    return records, account


def load_previous():
    try:
        with open(JSON_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def collect(names):
    with open(os.path.join(HERE, 'machines.json')) as f:
        machines = json.load(f)
    prev = load_previous().get('machines', {})
    now = dt.datetime.now()
    result = {}
    for key, machine in machines.items():
        if names and key not in names:
            if key in prev:
                result[key] = prev[key]
            continue
        ok, text = run_ssh(machine['ssh'], remote_script(machine))
        if ok:
            records, account = parse_output(machine, text, now)
            result[key] = {'title': machine['title'], 'read_at': now.strftime('%Y-%m-%d %H:%M'),
                           'fresh': True, 'login_help': machine['login_help'], 'account': account,
                           'queues': records}
            print(f'{key}: read {len(records)} queues', file=sys.stderr)
        else:
            old = prev.get(key, {})
            result[key] = {'title': machine['title'], 'read_at': old.get('read_at', ''), 'fresh': False,
                           'login_help': machine['login_help'], 'account': old.get('account', ''),
                           'queues': old.get('queues', []), 'error': text}
            print(f'{key}: not reachable ({text[:80]})', file=sys.stderr)
    return {'updated': now.strftime('%Y-%m-%d %H:%M'), 'machines': result}


def esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


JOB_LENGTHS = [('30 min', 30), ('1 hour', 60), ('4 hours', 240), ('2 days', 2880), ('14 days', 20160)]


def expected_wait(q):
    """Return (minutes, source) for a queue: the larger of the median of your recent waits and
    Slurm's estimate, so that one lucky history cannot hide a long queue; (None, '') when neither
    exists."""
    mine = int(statistics.median(q['recent_wait_minutes'])) if q.get('recent_wait_minutes') else None
    slurm = q.get('slurm_estimate_minutes')
    if mine is None and slurm is None:
        return None, ''
    if slurm is None or (mine is not None and mine >= slurm):
        return mine, 'my jobs'
    return slurm, 'Slurm'


def wait_class(mins):
    if mins is None:
        return ''
    if mins <= 30:
        return 'good'
    if mins <= 240:
        return 'ok'
    return 'bad'


def maint_text(mt):
    """Describe a maintenance window in one line, with times on the laptop's clock."""
    pct = f'{mt["coverage"] * 100:.0f}% of nodes'
    if mt['active']:
        return f'maintenance until {mt["end"]} ({pct})'
    return f'maintenance {mt["start"]} to {mt["end"]} ({pct}); starts in {fmt_minutes(mt["starts_in_minutes"])}'


def blocked_by_maintenance(q, need):
    """True when maintenance covers most of the queue and a job of `need` minutes would run into it."""
    mt = q.get('maintenance')
    if not mt or mt['coverage'] < MAINT_BLOCKS:
        return False
    return mt['active'] or mt['starts_in_minutes'] < need


def render(data):
    rows = []
    for key, m in data['machines'].items():
        short = m['title'].split(' (')[0]
        for q in m['queues']:
            w, src = expected_wait(q)
            rows.append({'machine': short, 'fresh': m['fresh'], 'read_at': m['read_at'], 'q': q,
                         'wait': w, 'src': src})
    rows.sort(key=lambda r: (r['q']['max_minutes'] if r['q']['max_minutes'] is not None else 10 ** 9,
                             r['wait'] if r['wait'] is not None else 10 ** 9))

    best = []
    for label, need in JOB_LENGTHS:
        ok = [r for r in rows if r['fresh'] and r['wait'] is not None
              and (r['q']['max_minutes'] is None or r['q']['max_minutes'] >= need)
              and not blocked_by_maintenance(r['q'], need)]
        ok.sort(key=lambda r: r['wait'])
        if ok:
            cells = ''.join(f'<td class="{wait_class(r["wait"])}">{esc(r["machine"])} {esc(r["q"]["name"])}: '
                            f'{fmt_minutes(r["wait"])}</td>' for r in ok[:3])
            best.append(f'<tr><td>up to {label}</td>{cells}</tr>')
        else:
            best.append(f'<tr><td>up to {label}</td><td colspan="3">no queue with a known wait</td></tr>')

    body = []
    for r in rows:
        q = r['q']
        cls = ' class="stale"' if not r['fresh'] else (f' class="{wait_class(r["wait"])}"' if r['wait'] is not None else '')
        gpus = f'{q["gpus_free"]} of {q["gpus_total"]}' if q['gpus_total'] else ''
        wait = f'{fmt_minutes(r["wait"])} ({r["src"]})' if r['wait'] is not None else 'unknown'
        mt = q.get('maintenance')
        name = esc(q['name']) + (f'<div class="maint">{esc(maint_text(mt))}</div>' if mt else '')
        body.append(f'<tr{cls}><td>{esc(q["max_job"])}</td><td>{esc(r["machine"])}</td><td>{name}</td>'
                    f'<td>{esc(q["gpu"])}</td><td class="num">{gpus}</td><td class="num">{q["waiting"]}</td>'
                    f'<td>{wait}</td><td class="muted">{esc(r["read_at"])}</td></tr>')

    windows = {}  # (machine, reservation) -> (window, queue names)
    for m in data['machines'].values():
        for q in m['queues']:
            mt = q.get('maintenance')
            if mt and m['fresh']:
                key = (m['title'].split(' (')[0], mt['name'])
                w = windows.setdefault(key, [mt, []])
                w[1].append(q['name'])
                if mt['coverage'] > w[0]['coverage']:
                    w[0] = mt
    maint_note = ''
    if windows:
        items = ''.join(f'<li><b>{esc(mach)}</b>: {esc(maint_text(mt))}.  Queues: {esc(", ".join(names))}.</li>'
                        for (mach, _), (mt, names) in windows.items())
        maint_note = (f'<h2>Maintenance</h2><ul class="maint-note">{items}</ul>'
                      '<p class="muted">Times are on this laptop\'s clock.  A queue drops out of "best bet" for '
                      'job lengths that would run into a window covering most of its nodes.</p>')

    stale = [m for m in data['machines'].values() if not m['fresh']]
    stale_note = ''
    if stale:
        items = ''.join(f'<li>{esc(m["title"])}: last read {esc(m["read_at"] or "never")}. To refresh: {esc(m["login_help"])}</li>'
                        for m in stale)
        stale_note = f'<p class="stale-note">Grey rows are stale.</p><ul class="stale-note">{items}</ul>'

    return f"""<title>GPU Queues</title>
<style>
:root {{ --bg: #fbfaf7; --fg: #1c1b18; --muted: #6b675e; --line: #d9d5cc; --head: #eeebe3; --stale: #a1452c;
        --good: #dff3e1; --ok: #fff3cf; --bad: #fbe0dc; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #1b1a18; --fg: #ebe8e1; --muted: #a8a399; --line: #3a3833; --head: #2a2926; --stale: #e07a5f; --good: #1f3b25; --ok: #3d3418; --bad: #46211c; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg: #1b1a18; --fg: #ebe8e1; --muted: #a8a399; --line: #3a3833; --head: #2a2926; --stale: #e07a5f; --good: #1f3b25; --ok: #3d3418; --bad: #46211c; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 14px/1.4 -apple-system, "Helvetica Neue", Arial, sans-serif; padding-block: 24px; padding-inline: 16px; margin: 0; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 22px 0 6px; }}
p {{ max-width: 75ch; margin: 6px 0; }}
.muted {{ color: var(--muted); }}
.wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
thead th {{ background: var(--head); font-weight: 600; }}
td.num {{ text-align: right; }}
tr.good td, td.good {{ background: var(--good); }}
tr.ok td, td.ok {{ background: var(--ok); }}
tr.bad td, td.bad {{ background: var(--bad); }}
tr.stale td {{ color: var(--muted); }}
.stale-note {{ color: var(--stale); }}
.maint, .maint-note {{ color: var(--stale); }}
.maint {{ font-size: 12px; }}
.legend {{ max-width: 80ch; padding-left: 20px; }}
.legend li {{ margin: 4px 0; }}
</style>
<h1>GPU Queues</h1>
<p class="muted">Updated {esc(data["updated"])}. Legend at the bottom.</p>
{maint_note}
<h2>Best bet by job length</h2>
<div class="wrap"><table>
<thead><tr><th>Job length</th><th>First choice</th><th>Second</th><th>Third</th></tr></thead>
<tbody>
{''.join(best)}
</tbody></table></div>
<h2>All queues, shortest job limit first</h2>
<div class="wrap"><table>
<thead><tr><th>Max job</th><th>Machine</th><th>Queue</th><th>GPU</th><th>GPUs free</th><th>Waiting</th><th>Expected wait</th><th>Read at</th></tr></thead>
<tbody>
{''.join(body)}
</tbody></table></div>
{stale_note}
<h2>Legend</h2>
<ul class="legend">
<li><b>Job length</b>: the longest run the queue allows. Pick the row that covers your job.</li>
<li><b>Expected wait</b>: the larger of two numbers, so a lucky history cannot hide a long queue. "my jobs" is the median wait of your own jobs in that queue over the last 30 days. "Slurm" is the scheduler's estimate for a typical test job (nothing is submitted); it assumes every running job uses its whole time limit, so it runs long.</li>
<li><b>Colors</b>: green, expected wait under 30 minutes. Yellow, under 4 hours. Red, longer. Grey, stale numbers from an earlier read.</li>
<li><b>GPUs free</b>: GPUs not in use on nodes that are up and not reserved. Free GPUs with jobs waiting usually means the waiting jobs need more than what is free.</li>
<li><b>Waiting</b>: jobs queued in that queue right now, from everyone.</li>
<li><b>Best bet</b>: the three queues with the shortest expected wait among those that allow your job length, machines you are logged into only.</li>
<li><b>Maintenance</b>: a Slurm maintenance reservation that is active or starts within {MAINT_HORIZON_DAYS} days and covers at least {MAINT_MIN_COVERAGE * 100:.0f}% of a queue's nodes. Nodes in an active reservation, or whose maintenance starts within {MAINT_SOON_MINUTES} minutes, do not count as free. Reservations longer than {MAINT_MAX_DAYS} days hold a few broken or test nodes and are not shown.</li>
</ul>
"""


def main():
    names = set(sys.argv[1:])
    os.makedirs(OUT_DIR, exist_ok=True)
    data = collect(names)
    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=1)
    with open(HTML_PATH, 'w') as f:
        f.write(render(data))
    for key, m in data['machines'].items():
        state = 'fresh' if m['fresh'] else 'STALE: ' + m['login_help']
        print(f'{m["title"]}: {state}')
        for q in m['queues']:
            print(f'  {q["name"]:28s} free {q["gpus_free"]:>3}/{q["gpus_total"]:<3} wait {q["waiting"]:>4} '
                  f'run {q["running"]:>4} max {q["max_job"]:10s} est {q["slurm_estimate"][:40]:40s} {q["recent_waits"]}')
            if q.get('maintenance'):
                print(f'  {"":28s} {maint_text(q["maintenance"])}')
    print('wrote', HTML_PATH)
    url = load_config().get('artifact_url')
    print('artifact_url', url if url else 'NOT SET (publish a new artifact and record its URL in ' + CONFIG_PATH + ')')


if __name__ == '__main__':
    main()
