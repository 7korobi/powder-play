<script lang="ts">
	import {
		GATE4_LIQUID_PARTICLE_DIAMETER,
		GATE4_LIQUID_PARTICLE_DIAMETER_MAX,
		GATE4_LIQUID_PARTICLE_DIAMETER_MIN
	} from './scenes/gate4.js';

	interface Props {
		value?: number;
		appliedValue: number;
		disabled?: boolean;
		onApply: () => void;
	}

	let {
		value = $bindable(GATE4_LIQUID_PARTICLE_DIAMETER),
		appliedValue,
		disabled = false,
		onApply
	}: Props = $props();
</script>

<fieldset class="particle-size">
	<legend>共通の液体粒径</legend>
	<label>
		<span>水・油の参照直径</span>
		<output>
			{value} units · 参照面積 {(value * value).toLocaleString()} units²
		</output>
		<input
			type="range"
			min={GATE4_LIQUID_PARTICLE_DIAMETER_MIN}
			max={GATE4_LIQUID_PARTICLE_DIAMETER_MAX}
			step="1"
			bind:value
			disabled={disabled}
			aria-label="水と油の共通粒子直径"
			aria-valuetext={`${value} world units`}
		/>
	</label>
	<p>
		両ページで同じ値を使用します。面積・液体質量・表示径を直径から導出し、粒子数と配置間隔は変えません。
		適用すると同じ seed でシーンを再初期化します。
	</p>
	<button
		type="button"
		disabled={disabled || value === appliedValue}
		onclick={onApply}
	>
		粒径を適用してシーンを再初期化
	</button>
</fieldset>

<style>
	.particle-size {
		min-width: 0;
		margin: 0;
		padding: 0.9rem;
		border: 1px solid #34465d;
		border-radius: 0.8rem;
		background: #0c1725;
	}

	legend {
		padding: 0 0.3rem;
		color: #dce9f8;
		font-weight: 700;
	}

	label {
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 0.5rem 0.8rem;
		align-items: center;
	}

	label span {
		color: #d6e0ed;
		font-size: 0.86rem;
	}

	output {
		color: #8bd8f3;
		font-size: 0.84rem;
		font-variant-numeric: tabular-nums;
	}

	input {
		grid-column: 1 / -1;
		width: 100%;
		accent-color: #55c4ee;
	}

	p {
		margin: 0.6rem 0;
		color: #9caec3;
		font-size: 0.78rem;
		line-height: 1.5;
	}

	button {
		width: 100%;
		padding: 0.55rem 0.7rem;
		border: 1px solid #45617e;
		border-radius: 0.55rem;
		background: #172b40;
		color: #e7eef8;
		font: inherit;
		font-size: 0.82rem;
		font-weight: 650;
		cursor: pointer;
	}

	button:disabled {
		opacity: 0.55;
		cursor: not-allowed;
	}
</style>
