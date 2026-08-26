# 完整物理评估

本目录定义候选线圈的接受路径。快速原生分数用于筛选与优化；最终接受必须重新选择本样本的拟合域和外层面，并通过标准表面求解与独立检查。

## 依赖与输入

- 已构建的 `libstellarator_gpu.so`（或平台对应动态库）；
- 安装 `simsopt`；最终阶段如需平衡计算再安装 DESC；
- 线圈 JSON，含 `raw.x/y/z/current/current_unit` 和 `nfp`；
- 独立输出目录。不要复用其他样本的 `a`、`s_edge` 或中间 NPZ。

## 1. 磁轴与源 $s$

对多个候选拟合半径 `a` 分别运行：

```bash
stellcoilopt-eval \
  --case-file examples/01.json --key raw --current-unit MA \
  --a 0.05 \
  --output-dir runs/case_a005
```

比较独立验证的角度 P95/L2、训练域覆盖和外层射线根。选定的 `run-dir` 必须包含 `summary.json` 和 `psi_model.npz`。

## 2. 拟合 $\alpha$

对本样本选定的 `s_edge` 运行 GPU-ray 体采样和 QR：

```bash
python scripts/fit_alpha.py \
  --run-dir runs/case_a005 \
  --case-file examples/01.json --case-key raw --current-unit MA \
  --s-edge 0.16 --orders 12:12:16 \
  --out-dir runs/case_alpha \
  --gpu-lib gpu_backend/build_native_score/libstellarator_gpu.so
```

检查固定训练/验证点预算、磁通单调性、 $\alpha$ 独立验证残差、法向场和 $1+\lambda_\theta$ 。

## 3. 构造 $\alpha+\nu$ 候选面

```bash
python scripts/fit_nu.py \
  --run-dir runs/case_a005 \
  --case-file examples/01.json \
  --alpha-dir runs/case_alpha \
  --alpha-fit alpha_fit_L12_M12_N16.npz \
  --s-edge 0.16 --rho-values 0.5,0.6,0.7,0.8,0.9,1.0 \
  --nu-orders 4,8,12 --save-surfaces \
  --output-dir runs/case_nu \
  --gpu-lib gpu_backend/build_native_score/libstellarator_gpu.so
```

每个 `rho` 输出 `surfaces/rho_<value>_alpha_nu.npz`。排除 Jacobian 变号、谱面退化和独立残差异常的初值。

## 4. 标准 LS/Newton

对各候选并行运行，避免一个外层困难面阻塞其他面：

```bash
python scripts/solve_boozer_surface.py \
  --case-file examples/01.json --case-key raw --current-unit MA \
  --run-dir runs/case_a005 \
  --surface-npz runs/case_nu/surfaces/rho_0p8_alpha_nu.npz \
  --output-dir runs/candidate_rho_0p8 \
  --gpu-lib gpu_backend/build_native_score/libstellarator_gpu.so
```

只有 `boozer_standard.npz` 表示该候选通过求解器和独立稠密网格检查；`boozer_rejected.npz` 保留失败证据，不能进入下游。

## 5. 选择最大连续可接受面

整理每个候选的 `summary.json` 后运行：

```bash
python evaluation/full_physical/select_largest_standard_surface.py \
  --candidate-root runs/candidates \
  --output runs/selected_surface.json
```

选择器要求 `accepted_for_downstream=true`，并沿目标 `s` 检查包围体积单调增加。外层候选即使 Newton 形式收敛，只要体积回落到内分支也会被拒绝。

## 6. Poincare、几何输出与 DESC

```bash
python scripts/evaluate_surface.py \
  --case-file examples/01.json --current-unit MA \
  --surface-npz runs/selected/boozer_standard.npz \
  --output-dir runs/full_evaluation
```

最终报告应同时检查：

- 独立稠密 Boozer 残差和法向场分位数；
- Poincare 嵌套；
- 面的环向绕数、法向非退化和坐标 Jacobian；
- QA/QH/QP 表面误差；
- DESC 边界一致性、嵌套性、力残差分布和求解状态。

DESC 到达迭代上限但残差下降不等于求解器收敛；`success`、残差、嵌套性和 Jacobian 必须分别报告。
