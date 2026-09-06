# LinkedIn drafts

Three versions. LinkedIn's feed truncates at roughly 200 characters, so the
first two lines have to carry the click — each draft is built that way.

**Links** (LinkedIn deprioritises posts with links in the body, so put the demo
link in the *first comment* and mention it in the post):

- Demo: https://huggingface.co/spaces/NagaYu/molt
- Code: https://github.com/NagaYu/molt
- Projectors: https://huggingface.co/NagaYu/molt-kv-projectors-qwen2.5
- Measurements: https://huggingface.co/datasets/NagaYu/molt-benchmark-results

**Attach `figures/demo.gif`.** LinkedIn autoplays GIFs, and the colour changing
mid-sentence is the entire pitch in three seconds.

---

## 1 · English

> Your phone takes memory back mid-answer. Your local LLM either dies or starts
> over.
>
> I spent a while asking whether there is a third option, and there is: move the
> running generation onto a smaller model **between two tokens**, and carry the
> KV cache with it.
>
> The paragraph in the clip is written by three different models. The colour
> behind each token is which one. The memory budget is cut mid-sentence and
> nothing restarts.
>
> Measured on a Qwen2.5 ladder (1.5B fp32 → int8 → 0.5B), CPU, five prompts,
> identical memory-pressure trace for every arm:
>
> ▸ always-large — reclaimed by the OS in 5 of 5 runs
> ▸ the same code allowed to switch only *between requests* — also 5 of 5
> ▸ carrying the cache mid-stream — 0, with the switch costing 900 ms against
>   3,615 ms for discard-and-re-prefill
>
> The detail that nearly sank it: HuggingFace caches attention keys *after*
> rotary embedding. Two models with different head dimensions therefore hold
> their caches in different rotational bases, and no position-independent matrix
> can map one onto the other — the map you would need depends on each token's
> absolute position. Un-rotate at the source's angles, project, re-rotate at the
> destination's, and a single small matrix per layer suddenly works.
>
> I have also published the parts that argue *against* the design, because an
> ablation table that only confirms its author is not an ablation table:
>
> ▸ One mechanism I was proud of — recomputing the destination's top layers
>   natively — cost 31× more per switch for a 2% smoother seam. Off by default.
> ▸ My first continuity metric measured nothing. Step-to-step divergence
>   saturates at ln 2 on real text, so the signal sat below the floor.
> ▸ A deliberately naive baseline *beat* the real method on the replacement
>   metric while producing 8× worse perplexity. Bounded divergences reward
>   blurring; the naive map made attention 11.4× flatter. Two metrics, read
>   together, or you get fooled.
>
> Code, the fitted maps, every measurement, and an interactive demo — all open,
> links in the comments.
>
> If you have shipped a local model on a phone and fought the OS for memory, I
> would genuinely like to hear how you handled it.
>
> #OnDeviceAI #EdgeAI #LLM #MachineLearning #Inference #OpenSource

---

## 2 · 日本語

> スマホがメモリを取り返しに来ると、ローカルLLMは死ぬか、最初からやり直すかしかない。
>
> 三つ目の選択肢があるか試していました。あります。走っている生成を**トークンと
> トークンの間で**小さいモデルに移し、KVキャッシュを連れて行く。
>
> 動画の段落は3つのモデルが書いています。トークンの背景色がどのモデルかを示していて、
> メモリ予算は文の途中で切られています。やり直しは起きていません。
>
> Qwen2.5のラダー(1.5B fp32 → int8 → 0.5B)、CPU、プロンプト5本、全条件で同一の
> メモリ圧トレース:
>
> ▸ 常に大きいモデル … 5回中5回、OSに回収された
> ▸ 同じコードでリクエスト境界でのみ切替 … これも5/5回収
> ▸ 生成途中でキャッシュを持って移動 … 0。切替コストは900ms(捨てて再prefillだと3,615ms)
>
> 危うく破綻した点。HuggingFaceはアテンションのキーを**RoPE適用後に**キャッシュします。
> head_dimが違うモデル同士はキャッシュが別の回転基底にあり、位置に依存しない行列では
> 写像できません。必要な写像が各トークンの絶対位置に依存してしまうからです。
> 元モデルの角度で逆回転し、射影し、先モデルの角度で再回転する。これで層あたり
> 小さな行列ひとつで足りるようになりました。
>
> 設計に**不利な**結果も併せて公開しています。自分の設計を肯定するだけの
> アブレーション表は、アブレーション表ではないので:
>
> ▸ 気に入っていた機構(移行先の上位層をネイティブ再計算する)は、継ぎ目2%の改善に
>   31倍のコストがかかりました。既定でオフにしました。
> ▸ 最初の連続性指標は何も測っていませんでした。実テキストでは step-to-step の
>   ダイバージェンスが ln2 に飽和し、信号が床下に沈みます。
> ▸ わざと素朴に作ったベースラインが、差し替えた指標では本手法に**勝ちました**。
>   perplexityは8倍悪いのに。有界なダイバージェンスはぼやけた分布を優遇します。
>   素朴な写像は注意分布を11.4倍平坦化していました。指標は2つ読まないと騙されます。
>
> コード、学習済みの射影、全測定値、インタラクティブなデモ、すべて公開しています。
> リンクはコメント欄に。
>
> スマホでローカルモデルを載せてOSとメモリを取り合った経験のある方、どう対処したか
> ぜひ聞かせてください。
>
> #オンデバイスAI #エッジAI #LLM #機械学習 #推論 #OSS

---

## 3 · Bilingual (Japanese first, English below)

Use this for a Japan-based audience that includes non-Japanese colleagues. Keep
the Japanese complete and the English tighter — a full duplicate reads as
padding.

> スマホがメモリを取り返しに来ると、ローカルLLMは死ぬか最初からやり直すかしかない。
> 三つ目の選択肢を作りました。走っている生成を**トークンとトークンの間で**小さい
> モデルに移し、KVキャッシュを連れて行く。
>
> 動画の段落は3つのモデルが書いています。背景色がどのモデルか。予算は文の途中で
> 切られ、やり直しは起きていません。
>
> Qwen2.5ラダー、CPU、全条件同一のメモリ圧トレース:
> ▸ 常に大きいモデル … 5/5 でOSに回収
> ▸ リクエスト境界でのみ切替 … これも 5/5 回収
> ▸ 生成途中で移動 … 0。切替900ms(再prefillなら3,615ms)
>
> 一番効いた細部は、HuggingFaceがキーを**RoPE適用後に**キャッシュすること。
> head_dimが違えば別の回転基底にあり、位置非依存の行列では写像できません。
> 逆回転→射影→再回転で、層あたり小さな行列ひとつに収まりました。
>
> 設計に不利な結果も公開しています。上位層の再計算は継ぎ目2%改善に31倍のコスト、
> 最初の連続性指標は ln2 に飽和して無意味、素朴なベースラインが指標では勝つのに
> perplexityは8倍悪い(有界なダイバージェンスはぼやけを優遇する)。
>
> ——
>
> **In English.** When a device reclaims memory mid-answer, a local LLM either
> gets killed or restarts. Molt does neither: it moves the running generation to
> a smaller model *between two tokens*, carrying the KV cache across. Same
> pressure trace, five prompts: always-large reclaimed 5/5, switching only
> between requests 5/5, mid-stream migration 0 — at 900 ms per switch against
> 3,615 ms for a re-prefill.
>
> The load-bearing detail: HF caches keys *after* RoPE, so models with different
> head dimensions hold their caches in different rotational bases and no
> position-independent matrix maps between them. Un-rotate, project, re-rotate.
>
> The results that argue against the design are published too — including a
> naive baseline that beats the real method on one metric while being 8× worse
> on perplexity, because bounded divergences reward blurring.
>
> Open source. Links in the comments.
>
> #OnDeviceAI #EdgeAI #LLM #機械学習 #OpenSource

---

## First comment (all versions)

> Demo (press Play — it is a replay of a real recorded session, no models to
> load): https://huggingface.co/spaces/NagaYu/molt
>
> Code: https://github.com/NagaYu/molt
> Fitted projectors: https://huggingface.co/NagaYu/molt-kv-projectors-qwen2.5
> Every measurement: https://huggingface.co/datasets/NagaYu/molt-benchmark-results
>
> Honest limits, up front: the worst single token barely improves on a cold
> start — that is model *loading*, which the restart baseline pays too. The
> transplanted cache also leaves a measurably rougher seam than a re-prefill.
> Both are in the README with numbers.

---

## Notes on why these are written this way

- **The hook is a failure the reader has lived through**, not the technique.
  "Your phone takes memory back mid-answer" lands with anyone who has shipped a
  local model; "mid-generation KV cache transplant" does not.
- **The negative results are in the post, not hidden in the repo.** On LinkedIn
  this is unusual enough to be memorable, and it pre-empts the one comment that
  would otherwise dominate the thread.
- **No superlatives, no "excited to announce".** The numbers are strong enough
  to carry themselves, and hedging language would make a reader assume they are
  not.
- **The closing question is real.** Asking people who have fought jetsam how
  they handled it invites the exact audience worth reaching, and LinkedIn's
  ranking rewards comment threads over reactions.
- **Do not post all three.** Pick one by audience: English for a global feed,
  Japanese for a domestic one, bilingual only if your feed is genuinely mixed.
