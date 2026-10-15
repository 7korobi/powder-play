<script lang="ts">
	const labSections = [
		{
			id: 'particle-manipulation',
			title: '粒子操作',
			description: 'WebGPU 上で粒子を選択・操作する基礎ラボ。',
			pages: [
				{
					href: '/labs/mpm',
					title: 'GPU 粒子操作ラボ',
					label: '操作プロトタイプ · Gate 0',
					description:
						'合成した流体風・ゲル風クラスターの選択、ドラッグ、複製など、GPU 粒子操作の基礎を試します。'
				}
			]
		},
		{
			id: 'mpm-solvers',
			title: 'MLS-MPM とソルバー比較',
			description: 'GPU 実装、同じ式の CPU 数値参照、LiquidFun の挙動を見比べるラボ。',
			pages: [
				{
					href: '/labs/mpm/kernels',
					title: 'GPU 2D MLS-MPM サンドボックス',
					label: 'WebGPU · Gate 1–4',
					description:
						'水・油・小石・破断固体を GPU MLS-MPM で動かし、計算カーネルや性能・挙動を調べます。'
				},
				{
					href: '/labs/mpm/reference',
					title: 'CPU MLS-MPM 数値参照',
					label: 'CPU · 同じ MLS-MPM 方程式',
					description:
						'GPU 版と同じ物理モデルを CPU で実行し、数値的な差や処理時間を比較します。'
				},
				{
					href: '/labs/mpm/liquidfun',
					title: 'LiquidFun WASM 挙動リファレンス',
					label: 'LiquidFun · Box2D/WASM',
					description:
						'別系統の物理エンジンにおける流体・接触・小石の動きを観察し、GPU 版の挙動目標を検討します。'
				}
			]
		}
	] as const;
</script>

<svelte:head>
	<title>Labs sitemap · powder-play</title>
	<meta
		name="description"
		content="powder-play の labs 配下にあるすべての実験ページへのサイトマップ"
	/>
</svelte:head>

<main class="page">
	<header class="page-header">
		<p class="eyebrow">POWDER-PLAY · EXPERIMENTS</p>
		<h1>Labs</h1>
		<p class="intro">
			実験・比較用ページの一覧です。ここから labs 配下の各ページへ移動できます。
			CPU MLS-MPM は GPU 版の数値参照、LiquidFun は異なる物理エンジンの挙動参照として、それぞれ役割を分けています。
		</p>
	</header>

	<nav class="sitemap" aria-label="Labs sitemap">
		{#each labSections as section}
			<section class="lab-section" aria-labelledby={section.id}>
				<div class="section-heading">
					<h2 id={section.id}>{section.title}</h2>
					<p>{section.description}</p>
				</div>
				<ul class="page-list">
					{#each section.pages as page}
						<li>
							<a class="page-card" href={page.href}>
								<span class="page-card-header">
									<strong>{page.title}</strong>
									<span class="page-label">{page.label}</span>
								</span>
								<span class="page-description">{page.description}</span>
								<code class="page-path">{page.href}</code>
							</a>
						</li>
					{/each}
				</ul>
			</section>
		{/each}
	</nav>

	<footer>
		<a href="/">← メイン画面へ戻る</a>
	</footer>
</main>

<style>
	:global(body) {
		margin: 0;
		background: #0b111b;
		color: #e7eef8;
		font-family:
			Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
	}

	.page {
		box-sizing: border-box;
		width: min(100%, 1120px);
		margin: 0 auto;
		padding: clamp(1rem, 4vw, 3rem);
	}

	.page-header {
		margin-bottom: 1.5rem;
	}

	.eyebrow {
		margin: 0 0 0.45rem;
		color: #7bd7f5;
		font-size: 0.75rem;
		font-weight: 800;
		letter-spacing: 0.12em;
	}

	h1,
	h2,
	p {
		margin-top: 0;
	}

	h1 {
		margin-bottom: 0.55rem;
		font-size: clamp(2rem, 5vw, 3.5rem);
		line-height: 1.1;
	}

	h2 {
		margin-bottom: 0.35rem;
		font-size: 1.2rem;
	}

	.intro {
		max-width: 78ch;
		margin-bottom: 0;
		color: #bdcada;
		line-height: 1.65;
	}

	.sitemap {
		display: grid;
		gap: 1.1rem;
	}

	.lab-section {
		border: 1px solid #26374d;
		border-radius: 14px;
		padding: clamp(1rem, 2.5vw, 1.5rem);
		background: #121c2a;
	}

	.section-heading {
		margin-bottom: 1rem;
	}

	.section-heading p {
		margin-bottom: 0;
		color: #aebed1;
		line-height: 1.5;
	}

	.page-list {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 17rem), 1fr));
		gap: 0.75rem;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.page-card {
		display: grid;
		height: 100%;
		box-sizing: border-box;
		gap: 0.7rem;
		border: 1px solid #30445c;
		border-radius: 10px;
		padding: 1rem;
		background: #0d1823;
		color: inherit;
		text-decoration: none;
		transition:
			border-color 120ms ease,
			background-color 120ms ease,
			transform 120ms ease;
	}

	.page-card:hover {
		transform: translateY(-2px);
		border-color: #55bddc;
		background: #14263a;
	}

	.page-card:focus-visible,
	footer a:focus-visible {
		outline: 3px solid #7bd7f5;
		outline-offset: 3px;
	}

	.page-card-header {
		display: grid;
		gap: 0.45rem;
	}

	.page-card-header strong {
		color: #eaf6ff;
		font-size: 1rem;
	}

	.page-label {
		width: fit-content;
		border: 1px solid #35536b;
		border-radius: 999px;
		padding: 0.2rem 0.55rem;
		color: #a8dff0;
		font-size: 0.75rem;
	}

	.page-description {
		color: #b7c7d9;
		font-size: 0.9rem;
		line-height: 1.5;
	}

	code {
		color: #8be0fa;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.85rem;
	}

	.page-path {
		align-self: end;
		overflow-wrap: anywhere;
	}

	footer {
		margin-top: 1.5rem;
	}

	footer a {
		color: #8be0fa;
		text-underline-offset: 0.18em;
	}
</style>
