
from dotenv import load_dotenv
# ─── CONFIG ─────────────────────────────────────────────────────────
load_dotenv() 

TEST_CSV       = 'kaggledatasetrawfiles/linkdin_Job_data_test.csv'
OUTPUT_JSON    = 'data/eval_samples_metaphors.json'
MAX_RETRIES    = 3
SIM_THRESHOLD  = 0.75
MAX_CORRECT    = 3
MAX_INCORRECT  = 3

# ─── RETRY HELPER ────────────────────────────────────────────────────
def parse_wait(msg: str) -> float:
    m = re.search(r"(\d+)m(\d+\.?\d*)s", msg)
    if m:
        mins, secs = float(m[1]), float(m[2])
        return mins*60 + secs + 5
    return 30

def safe_generate(length, language, tag) -> str | None:
    """Wrap generate_post() to handle rate limits and 5xx errors."""
    for attempt in range(1, MAX_RETRIES+1):
        try:
            return generate_post(length, language, tag).strip()
        except RateLimitError as e:
            wait = parse_wait(e.response['error']['message'])
            print(f"[RateLimit] attempt {attempt}/{MAX_RETRIES}, sleeping {int(wait)}s…")
            time.sleep(wait)
        except InternalServerError as e:
            backoff = 5 * attempt
            print(f"[5xx] attempt {attempt}/{MAX_RETRIES}, retrying in {backoff}s…")
            time.sleep(backoff)
        except Exception as e:
            print(f"[Error] unexpected: {e!r}. Skipping sample.")
            return None
    print("⚠️  All retries failed—skipping this sample.")
    return None

# ─── LOAD TEST SET ──────────────────────────────────────────────────
df = pd.read_csv(TEST_CSV, usecols=['post_id','post_text'])
samples = []

for _, row in df.iterrows():
    input_txt = row['post_text'].strip()
    pred = safe_generate(length='Short', language='English', tag='Job Search')
    if not pred:
        continue

    sim = difflib.SequenceMatcher(None, pred, input_txt).ratio()
    samples.append({
        'post_id':      row['post_id'],
        'input_text':   input_txt,
        'prediction':   pred,
        'similarity':   round(sim,3)
    })

# ─── SELECT TOP & BOTTOM 3 ───────────────────────────────────────────
samples_sorted = sorted(samples, key=lambda x: x['similarity'], reverse=True)
correct   = samples_sorted[:MAX_CORRECT]
incorrect = samples_sorted[-MAX_INCORRECT:]

# ─── DUMP RESULTS ───────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump({'correct_samples': correct, 'incorrect_samples': incorrect},
              f, indent=4, ensure_ascii=False)

print(f"Picked {len(correct)} correct & {len(incorrect)} incorrect samples → {OUTPUT_JSON}")