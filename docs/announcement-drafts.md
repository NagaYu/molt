# Announcement drafts

Copy-paste ready. Every number below is from `benchmarks/results/summary.json`
— if you re-run the benchmark, re-check these before posting.

**Fill in before use:** `<REPO>` = the GitHub URL, `<REPORT>` = the results page.

**Posting order that works:** r/LocalLLaMA first (most on-target, most forgiving
of a prototype, and their questions will sharpen the framing) → Hacker News a
day or two later with the answers already in hand → X/Twitter last, linking the
discussion that already happened.

---

## 1 · Hacker News (Show HN)

**Title** (80 char limit; the first one is the strongest — it is concrete,
slightly surprising, and does not oversell):

> Show HN: Molt – swap the LLM mid-sentence and carry the KV cache across

Alternates:
- `Show HN: Molt – an LLM that changes model between two tokens when memory runs out`
- `Show HN: Molt – mid-generation KV cache transplant between different-sized models`

**URL:** `<REPO>`

**First comment** (post immediately; HN readers look for it):

> Author here. The narrow question I wanted to answer: when a phone or laptop
> takes memory back mid-answer, does an on-device LLM have to either die or
> start over?
>
> Molt does neither. It moves the running generation onto a smaller model
> *between two tokens* and carries the KV cache with it. On a Qwen2.5 ladder
> (1.5B fp32 → 1.5B int8 → 0.5B fp32) under the same memory-pressure trace:
>
> - always-large: reclaimed by the OS in 5 of 5 runs
> - the same code restricted to switching only *between requests*: also 5 of 5
> - Molt: 0, with the switch costing 900 ms against 3615 ms for
>   discard-and-re-prefill. Isolating the KV work (both models already warm)
>   it is 4.4–5.1× faster and avoids 74–79% of the FLOPs.
>
> The detail I did not expect to matter as much as it did: HF caches keys
> *after* RoPE, so two models with different head_dim have their caches in
> different rotational bases. No position-independent matrix can map one onto
> the other — the required map would depend on each token's absolute position.
> The fix is to sandwich the learned projection between an un-rotation at the
> source's angles and a re-rotation at the destination's. Without that the
> whole thing is incoherent, and I spent a while convinced the idea just did
> not work.
>
> Second thing worth stealing: a transplant needs the source model's RoPE
> parameters (a few KB of inv_freq), not its weights. So you can evict the
> outgoing model *before* loading the incoming one, and peak memory is
> max(src, dst) rather than src + dst. That matters, because the moment you
> need to migrate is exactly the moment you cannot afford both.
>
> Things that did **not** work, which are in the README because they change how
> you should use it:
>
> - Selective top-k layer recompute (recompute the destination's last k layers
>   natively instead of projecting them) cost 31× more per switch for a 2%
>   smoother seam on this ladder. Ship it off by default.
> - My first continuity metric was worthless. Step-to-step JSD between
>   consecutive token distributions saturates at ln 2 ≈ 0.693 on real text
>   (measured steady state 0.65–0.69), so a migration's contribution is below
>   the noise floor and the "excess" came out negative.
> - A naive truncated-identity map *beats* the learned projection on the
>   replacement metric (0.192 vs 0.220) while producing perplexity 24.1 against
>   3.1. JSD is bounded and rewards blurring: the naive map induces attention
>   11.4× flatter than the destination model's own, which is exactly the blur
>   that flatters a bounded divergence. Two metrics, read together, or you get
>   fooled.
>
> Honest limits: the worst single token barely improves in a cold-start run,
> because it is dominated by *loading* the destination model, which both
> strategies pay. What Molt removes is the re-prefill on top. The transplanted
> cache also produces a measurably rougher seam than a re-prefill does — 0.220
> against a 0.039 floor. And the portable INT8 fallback (bitsandbytes is
> CUDA-only) trades decode speed for memory, ~4× slower per token; a real kernel
> removes that and the memory result stands either way.
>
> Test suite is hermetic — `pytest` runs 72 tests on randomly-initialised tiny
> models in about a minute with no downloads, so you can check the mechanism
> without pulling 6 GB.

---

## 2 · r/LocalLLaMA

**Title:**

> Molt: swapping the model mid-sentence and carrying the KV cache across, so an
> OOM doesn't kill the generation

**Body:**

> If you have shipped a local model on a phone or a laptop you know the failure:
> a long answer is streaming, the user opens the camera, the OS wants memory
> back, and your process gets reclaimed mid-sentence. The usual mitigations are
> both bad — run the small model always (you pay on every request for a rare
> event) or restart the request on a smaller model (multi-second stall while the
> prompt is re-read).
>
> Molt takes a third option: move the running generation to a smaller model
> **between two tokens**, taking the KV cache with it.
>
> [GIF]
>
> That paragraph is written by three different models. The colour behind each
> token is the one that produced it. The budget was cut mid-sentence; nothing
> restarted.
>
> **Numbers** (Qwen2.5-1.5B fp32 / 1.5B int8 / 0.5B fp32, CPU, identical
> pressure trace for every arm, 5 prompts each):
>
> | | reclaimed by OS | switch cost | accuracy | judge agreement |
> |---|--:|--:|--:|--:|
> | always 1.5B | **5/5** | — | 0.00 | — |
> | always 0.5B | 0/5 | — | 0.67 | 0.750 |
> | restart on pressure | 0/5 | 3615 ms | 1.00 | 0.975 |
> | **Molt** | **0/5** | **900 ms** | 1.00 | 0.817 |
> | switch only between requests | **5/5** | — | 0.00 | — |
>
> With both rungs already warm, the KV work alone is 4.4–5.1× faster than
> re-prefilling and avoids 74–79% of the FLOPs.
>
> **How the cache crosses.** Three mechanisms, all fitted offline by closed-form
> ridge regression on a few thousand tokens of generic text (no SGD, seconds to
> fit):
>
> 1. A learned per-layer affine map on K and V — wrapped in a RoPE
>    un-rotate/re-rotate sandwich. This one is load-bearing: HF caches keys
>    *after* RoPE, so two models with different head_dim keep their caches in
>    different rotational bases and no position-independent matrix can map
>    between them. De-rotate, project, re-rotate.
> 2. A diagonal scale re-alignment when the two rungs share weights and differ
>    only in precision (fp32 → int8). Held-out residual 0.013 on keys.
> 3. Selective top-k native recompute of the destination's last layers.
>
> **Ablations, including the ones that argue against the design**, because a
> table that only confirms your design is not a table:
>
> - Remove the learned projection → perplexity 3.09 → 24.14. Essential.
> - Remove top-k recompute → seam 2% worse, switch **31× cheaper** (900 → 29 ms).
>   Not worth it on this ladder; use `--recompute-top-k 0`.
> - Restrict to request boundaries → reclaimed 5/5. The mid-stream ability is
>   the whole thing.
>
> **Caveats up front:** the portable INT8 path is dequant-on-use because
> bitsandbytes is CUDA-only, so that rung buys memory and costs ~4× decode
> speed. The worst single token doesn't improve much in a cold start — that's
> model *loading*, which the restart baseline pays too. And the transplanted
> cache leaves a measurably rougher seam than a re-prefill (0.220 vs a 0.039
> floor); most of that is the value projection, which is the weakest map.
>
> Apache-style MIT, CPU-first, `pytest` runs the whole thing in a minute with no
> downloads. Would genuinely like to hear from anyone who has fought jetsam
> with a local model — the pressure-source interface is one file and I would
> like it wired to real platform signals.
>
> Code: `<REPO>` · Full results: `<REPORT>`

---

## 3 · X / Twitter (English thread)

**1/** *(attach `figures/demo.gif`)*

> Your phone takes memory back mid-answer. Your local LLM either dies or starts
> over.
>
> Molt does a third thing: it swaps the model *between two tokens* and carries
> the KV cache across.
>
> This paragraph is written by three different models. The colour is which one.

**2/**

> Same memory-pressure trace, five prompts, Qwen2.5 1.5B → int8 → 0.5B on CPU:
>
> always-large — reclaimed by the OS 5/5
> switch only between requests — reclaimed 5/5
> Molt — 0
>
> Switch cost 900 ms vs 3615 ms for discard-and-re-prefill.

**3/**

> The detail that nearly killed the idea: HuggingFace caches keys AFTER RoPE.
>
> Two models with different head_dim hold their caches in different rotational
> bases. No position-independent matrix maps one onto the other — the map you'd
> need depends on each token's absolute position.

**4/**

> Fix: sandwich the learned projection between an un-rotation at the source's
> angles and a re-rotation at the destination's.
>
> Suddenly a single small matrix per layer works, and you can fit it in seconds
> by ridge regression on a few thousand tokens.

**5/**

> Second thing worth stealing: a transplant needs the source model's RoPE
> parameters — a few KB — not its weights.
>
> So you evict the outgoing model BEFORE loading the incoming one. Peak memory
> is max(src, dst), not src + dst.
>
> Which matters, because that's the moment you can't afford both.

**6/**

> Things that didn't work, in the README because they change how you'd use it:
>
> • top-k layer recompute: 31× the switch cost for a 2% smoother seam. Off by
>   default.
> • my first continuity metric measured nothing — step-to-step JSD saturates at
>   ln2 on real text.

**7/**

> And a trap worth knowing: a naive truncated map BEATS the learned projection
> on JSD (0.192 vs 0.220) while producing 8× worse perplexity.
>
> JSD is bounded and rewards blurring. The naive map makes attention 11.4×
> flatter. Read two metrics or get fooled.

**8/**

> Honest limits: the worst single token barely improves cold, because that's
> model *loading* — both strategies pay it. The seam is rougher than a
> re-prefill's. The portable INT8 rung buys memory and costs decode speed.
>
> Code + full numbers: `<REPO>`

---

## 4 · X / Twitter（日本語スレッド）

**1/** *(GIF添付)*

> スマホがメモリを取り返しに来ると、ローカルLLMは死ぬか最初からやり直すかしかない。
>
> Molt は3つ目の選択肢を取ります。**トークンとトークンの間でモデルを乗り換え、
> KVキャッシュを持って行く。**
>
> この段落は3つのモデルが書いています。背景色がそれです。

**2/**

> 同一のメモリ圧トレース、プロンプト5本、Qwen2.5 1.5B → int8 → 0.5B、CPU:
>
> 常時大モデル … OSに5/5回収された
> リクエスト境界でのみ切替 … 5/5回収
> Molt … 0
>
> 切替コストは 900ms。キャッシュを捨てて再prefillすると 3615ms。

**3/**

> 危うく破綻した点。HuggingFace はキーを **RoPE適用後に** キャッシュします。
>
> head_dim が違うモデル同士はキャッシュが別の回転基底にあり、位置非依存の行列
> では写像できません。必要な写像が各トークンの絶対位置に依存してしまう。

**4/**

> 解法は、学習した射影を「元モデルの角度で逆回転 → 射影 → 先モデルの角度で
> 再回転」で挟むこと。
>
> これで層あたり小さな行列1つで済み、数千トークンのリッジ回帰で数秒で学習できます。

**5/**

> もう一つ持ち帰る価値のある点。移植に必要なのは**元モデルのRoPEパラメータ
> 数KBだけで、重みは要らない**。
>
> つまり移行先をロードする前に移行元を解放できる。ピークメモリが max(src,dst)
> であって src+dst ではない。両方抱える余裕が無いのが、まさに移行が必要な瞬間なので。

**6/**

> うまくいかなかったことも README に書いています。使い方が変わるので:
>
> ・上位k層の再計算 … 継ぎ目2%改善に31倍のコスト。既定でオフ
> ・最初の連続性指標は何も測っていなかった。実テキストでは step-to-step JSD が
>   ln2 に飽和する

**7/**

> 知っておく価値のある罠。素朴な切り詰め写像が JSD では学習射影に**勝ちます**
> (0.192 vs 0.220)。なのに perplexity は8倍悪い。
>
> JSD は有界で、ぼやけた分布ほど得をする。素朴写像は注意分布を11.4倍平坦化して
> いました。指標は2つ読まないと騙されます。

**8/**

> 正直な限界: コールドスタートでは最悪トークン遅延はほぼ改善しません。あれは
> モデルの**ロード**時間で、再prefill側も同じだけ払っています。継ぎ目も
> 再prefillより粗い。可搬INT8はメモリを買ってデコード速度を払っています。
>
> コードと全数値: `<REPO>`

---

## 5 · Zenn / Qiita 記事

**タイトル案**

> 生成の途中でLLMを乗り換える — KVキャッシュを異サイズモデル間で移植する

**リード**

> スマホやノートPCでローカルLLMを動かすと、必ずこの失敗に出会います。長い回答を
> ストリーミングしている最中にユーザーがカメラを開き、OSがメモリを返せと言い、
> プロセスが文の途中で回収される。
>
> 対処法は2つとも筋が悪い。常に小さいモデルを使う(稀な事象の保険として全リクエスト
> で品質を払う)か、小さいモデルでリクエストをやり直す(プロンプト再読み込みの間、
> 数秒のストールをユーザーが見る)か。
>
> Molt は3つ目を取ります。走っている生成を**トークンとトークンの間で**小さい
> モデルへ移し、KVキャッシュを連れて行く。

**構成**

1. 問題 — jetsam と、リクエスト境界での切替が構造的に間に合わない理由
2. 一番効いた細部 — RoPE適用後キャッシュと、逆回転サンドイッチ
3. `TierRef` の観察 — 移植に要るのは重みでなくRoPEパラメータ、だからピークが `max` で済む
4. 測った結果 — 4条件 + アブレーション表
5. **捨てた指標3つと、設計に不利なアブレーション2つ**
6. サービスとして動かす — SSE、tier をトークンごとに返す、拒否ではなく待たせる
7. 限界

> ※ 5節が記事の価値の中心です。日本語の技術記事で「自分の設計に不利な測定結果」を
> 先に出す例は少なく、そこが信頼になります。順番を後ろに下げないでください。

---

## 使い回せる一行

- 「トークンとトークンの間でモデルを乗り換え、KVキャッシュを連れて行く」
- "The animal walks out of the shell." (プロジェクト名の由来。英語圏ではこれが一番刺さります)
- "A transplant needs the source model's RoPE parameters, not its weights."
- 「この段落は3つのモデルが書いています」

## 想定質問と答え

**Q. vLLM/SGLang でいいのでは？**
サーバ側の弾性とは制約が違います。あちらはリクエスト間で捌けますが、こちらは
1つの回答の途中でメモリが消えます。README でもサーバ側キャッシュ共有と
重みストリーミングは明示的に範囲外にしています。

**Q. int8 がデコード4倍遅いなら意味がないのでは？**
これは移植機構の実証であって量子化カーネルの話ではありません。bitsandbytes が
CUDA専用なので可搬フォールバックを書いた結果です。CUDA の bnb や Apple の ANE
では消える制約で、メモリ側の結果はどちらでも成立します。

**Q. 品質は落ちるのでは？**
落ちます。継ぎ目は再prefillより粗い(0.220 vs 0.039)。ただし常時小モデルよりは
上です(正答率 1.00 vs 0.67、judge一致 0.817 vs 0.750)。表に全部あります。

**Q. Llama じゃないのはなぜ？**
Hub でゲートされているためです。Qwen2.5 は必要な構造的性質(hidden 1536 vs 896、
28層 vs 24層、head_dim 128 vs 64、トークナイザ共通)を全部持っているので、
量子化ルートと異サイズルートの両方を1つのラダーで試せます。
