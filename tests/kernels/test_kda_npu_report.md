# KDA 算子 NPU 测试覆盖报告

## 1. vLLM 中 KDA 的调用场景

KDA (Kimi Delta Attention) 在 vLLM 中由 `KimiDeltaAttention` 层（`vllm/model_executor/layers/kda.py`）使用，分为两条路径：

**Prefill 路径**（`num_prefills > 0`）调用 `chunk_kda`：
```python
# kda.py:419-432
chunk_kda(
    q=q, k=k, v=v, g=g1, beta=beta,
    initial_state=initial_state,        # 从 recurrent_state 索引，始终非 None；新请求对应行置零
    output_final_state=True,             # 始终 True，需更新 cache
    use_qk_l2norm_in_kernel=True,        # 始终 True
    cu_seqlens=non_spec_query_start_loc, # 始终非 None，varlen 模式
)
```

**Decode 路径**（`num_prefills == 0`）调用 `fused_recurrent_kda`：
```python
# kda.py:437-452
fused_recurrent_kda(
    q=q, k=k, v=v, g=g1, beta=beta,
    initial_state=recurrent_state,       # 原地更新（inplace_final_state=True）
    use_qk_l2norm_in_kernel=True,        # 始终 True
    cu_seqlens=non_spec_query_start_loc[:num_decodes+1],  # varlen
    ssm_state_indices=non_spec_state_indices_tensor,       # 非 None
)
```

此外，`KimiDeltaAttention.forward` 还调用了 `fused_kda_gate`（门控计算）和 `FusedRMSNormGated`（输出归一化），它们是独立于 KDA 核心算子的子模块。

---

## 2. 测试文件概览

| 测试文件 | 测试目标 | 对应 vLLM 路径 |
|----------|---------|---------------|
| `test_chunk_kda_npu.py` | `chunk_kda`（prefill 并行模式） | Prefill 路径 |
| `test_fused_recurrent_kda_npu.py` | `fused_recurrent_kda`（decode 递归模式） | Decode 路径 |

两者均采用朴素递归实现 (`naive_recurrent_kda`, float32) 作为参考基准，通过 RMSE 相对误差阈值判定是否通过。

### 通用验证指标

| 输出 | 阈值 | 说明 |
|------|------|------|
| `o` (注意力输出) | RMSE ratio < 0.005 | 相对误差 |
| `ht` (最终状态) | RMSE ratio < 0.005 | 相对误差 |

### 通用验证方法

- NaN 检测：断言 Triton 输出不含 NaN
- RMSE 相对误差：`rmse(ref - tri) / rmse(ref)`
- 绝对误差容差：`atol=1e-6`（绝对误差低于此值直接通过，跳过相对误差检查）

---

## 3. Prefill 路径：`test_chunk_kda_npu.py` — `chunk_kda`

### 3.1 参数化组合 (9 组)

| # | H | D | cu_seqlens | dtype | 测试 ID |
|---|---|---|------------|-------|---------|
| 1 | 32 | 128 | [0, 64] | float16 | H32-D128-cu[0, 64]-float16 |
| 2 | 32 | 128 | [0, 1024] | float16 | H32-D128-cu[0, 1024]-float16 |
| 3 | 32 | 128 | [0, 15] | float16 | H32-D128-cu[0, 15]-float16 |
| 4 | 32 | 128 | [0, 256, 512, 768, 1024] | float16 | H32-D128-cu[0, 256, 512, 768, 1024]-float16 |
| 5 | 32 | 128 | [0, 15, 100, 300, 1200] | float16 | H32-D128-cu[0, 15, 100, 300, 1200]-float16 |
| 6 | 64 | 128 | [0, 256, 500, 1000] | float16 | H64-D128-cu[0, 256, 500, 1000]-float16 |
| 7 | 32 | 128 | [0, 8192] | float16 | H32-D128-cu[0, 8192]-float16 |
| 8 | 32 | 128 | [0, 256, 500, 1000] | bfloat16 | H32-D128-cu[0, 256, 500, 1000]-bfloat16 |
| 9 | 32 | 128 | [0, 4096] | float16 | H32-D128-cu[0, 4096]-float16 |

### 3.2 已覆盖维度

| 维度 | 已覆盖值 | 覆盖说明 |
|------|---------|---------|
| **头数 H** | 32, 64 | 小头数 + 2x 头数 |
| **头维度 D** | 128 | vLLM 模型配置值 |
| **序列长度 T** | 15, 64, 256, 500, 512, 768, 1000, 1024, 1200, 4096, 8192 | 短序列到长序列 |
| **数据类型** | float16, bfloat16 | 两种 NPU 常用精度 |
| **序列模式** | 单序列、等长多序列、不等长多序列 | cu_seqlens 包含 1~4 条序列 |
| **初始状态** | 有 (h0 随机初始化) | `initial_state` 非 None |
| **输出最终状态** | 是 | `output_final_state=True` |
| **QK L2归一化** | 是 | `use_qk_l2norm_in_kernel=True` |
| **cu_seqlens (变长)** | 是 | 所有用例均传入 cu_seqlens |

### 3.3 与 vLLM 实际调用参数的对照

| 参数 | vLLM 实际值 | 测试覆盖 | 匹配 |
|------|-----------|---------|------|
| `use_qk_l2norm_in_kernel` | True | True | ✅ |
| `initial_state` | 非 None | 非 None | ✅ |
| `output_final_state` | True | True | ✅ |
| `cu_seqlens` | 非 None（varlen） | 非 None（varlen） | ✅ |
| `scale` | 默认 K**-0.5 | 默认 K**-0.5 | ✅ |
| K, V 维度 | K=V=head_dim | K=V=D | ✅ |

### 3.4 已覆盖的 chunk_kda 内部子算子

通过 `chunk_kda` 端到端调用，间接覆盖了以下子算子：

| 子算子 | 覆盖方式 |
|--------|---------|
| `chunk_local_cumsum` | chunk_kda_fwd 内部调用，varlen 模式 |
| `chunk_kda_scaled_dot_kkt_fwd` | intra-chunk K^T*K 计算，varlen 模式 |
| `solve_tril` | 下三角求解，varlen 模式 |
| `recompute_w_u_fwd` | w/u 重计算 + kg 生成，varlen 模式 |
| `chunk_gated_delta_rule_fwd_h` | 跨 chunk 状态递推 h，varlen 模式 |
| `chunk_gla_fwd_o_gk` | 最终输出 o 计算，varlen 模式 |
| `l2norm_fwd` | QK L2 归一化（在 chunk_kda 入口显式调用） |

---

## 4. Decode 路径：`test_fused_recurrent_kda_npu.py` — `fused_recurrent_kda`

### 4.1 Test 1: `test_fused_recurrent_kda` — 非原地 varlen 模式

验证输出 `o` 和每条序列的最终状态 `ht`，`inplace_final_state=False`。

| # | H | D | cu_seqlens | dtype | 测试 ID | 场景说明 |
|---|---|---|------------|-------|---------|---------|
| 1 | 32 | 128 | [0, 1] | float16 | H32-D128-cu[0, 1]-float16 | 单 token decode |
| 2 | 32 | 128 | [0, 1, 2, 3, 4] | float16 | H32-D128-cu[0, 1, 2, 3, 4]-float16 | 4 条单 token decode |
| 3 | 32 | 128 | [0, 1, 2, 3, 4, 5, 6, 7, 8] | float16 | H32-D128-cu[0, 1, 2, 3, 4, 5, 6, 7, 8]-float16 | 8 条单 token decode |
| 4 | 32 | 128 | [0, 16] | float16 | H32-D128-cu[0, 16]-float16 | 短序列多 token 递推 |
| 5 | 32 | 128 | [0, 8, 24] | float16 | H32-D128-cu[0, 8, 24]-float16 | 不等长多序列 |
| 6 | 32 | 128 | [0, 4, 8, 16] | float16 | H32-D128-cu[0, 4, 8, 16]-float16 | 不等长多序列 |
| 7 | 32 | 128 | [0, 64] | float16 | H32-D128-cu[0, 64]-float16 | 中等长度序列 |
| 8 | 64 | 128 | [0, 1, 2, 3, 4] | float16 | H64-D128-cu[0, 1, 2, 3, 4]-float16 | 不同头数 |
| 9 | 32 | 128 | [0, 1, 2, 3, 4] | bfloat16 | H32-D128-cu[0, 1, 2, 3, 4]-bfloat16 | BFloat16 |
| 10 | 32 | 128 | [0, 8, 24] | bfloat16 | H32-D128-cu[0, 8, 24]-bfloat16 | BFloat16 |

### 4.2 Test 2: `test_fused_recurrent_kda_decode_inplace` — 原地 decode + ssm_state_indices

模拟 vLLM 实际 decode 模式：`inplace_final_state=True`，使用 `ssm_state_indices` 索引状态槽位。验证原地状态更新正确性，以及 NULL 槽位（index 0）不被修改。

| # | H | D | N (序列数) | dtype | 测试 ID | 场景说明 |
|---|---|---|-----------|-------|---------|---------|
| 1 | 32 | 128 | 1 | float16 | H32-D128-N1-float16 | 单请求 decode |
| 2 | 32 | 128 | 4 | float16 | H32-D128-N4-float16 | 多请求 decode |
| 3 | 32 | 128 | 16 | float16 | H32-D128-N16-float16 | 较大批量 decode |
| 4 | 64 | 128 | 4 | float16 | H64-D128-N4-float16 | 不同头数 |
| 5 | 32 | 128 | 4 | bfloat16 | H32-D128-N4-bfloat16 | BFloat16 |

### 4.3 Test 3: `test_fused_recurrent_kda_fp32` — Float32 精度隔离

与 Test 1 相同场景，但使用 float32，隔离算法误差与低精度累积误差。

| # | H | D | cu_seqlens | 测试 ID | 场景说明 |
|---|---|---|------------|---------|---------|
| 1 | 32 | 128 | [0, 1] | H32-D128-cu[0, 1] | 单 token decode |
| 2 | 32 | 128 | [0, 1, 2, 3, 4] | H32-D128-cu[0, 1, 2, 3, 4] | 多条单 token decode |
| 3 | 32 | 128 | [0, 16] | H32-D128-cu[0, 16] | 短序列多 token |
| 4 | 32 | 128 | [0, 8, 24] | H32-D128-cu[0, 8, 24] | 不等长多序列 |

### 4.4 已覆盖维度

| 维度 | 已覆盖值 | 覆盖说明 |
|------|---------|---------|
| **头数 H** | 32, 64 | 小头数 + 2x 头数 |
| **头维度 D** | 128 | vLLM 模型配置值 |
| **序列长度 T** | 1, 4, 8, 16, 24, 64 | decode 为主（T=1），含短序列递推 |
| **数据类型** | float16, bfloat16, float32 | float32 用于隔离精度误差 |
| **序列模式** | 单 token decode、多 token 递推、不等长多序列 | |
| **inplace_final_state** | True, False | Test 2 用 True（vLLM 实际模式），Test 1/3 用 False |
| **ssm_state_indices** | 有（Test 2），无（Test 1/3） | Test 2 完全匹配 vLLM decode 调用模式 |
| **use_qk_l2norm_in_kernel** | True | 与 vLLM 一致 |
| **cu_seqlens (变长)** | 是 | 所有用例均传入 cu_seqlens |
| **NULL 槽位保护** | 是（Test 2） | 验证 index=0 的槽位不被修改 |

### 4.5 与 vLLM 实际调用参数的对照

| 参数 | vLLM 实际值 | 测试覆盖 | 匹配 |
|------|-----------|---------|------|
| `use_qk_l2norm_in_kernel` | True | True | ✅ |
| `initial_state` | 非 None（recurrent_state） | 非 None | ✅ |
| `inplace_final_state` | True | True（Test 2）/ False（Test 1/3） | ✅ |
| `cu_seqlens` | 非 None（varlen） | 非 None（varlen） | ✅ |
| `ssm_state_indices` | 非 None | 非 None（Test 2） | ✅ |
| `scale` | 默认 K**-0.5 | 默认 K**-0.5 | ✅ |

**Test 2 完全匹配 vLLM decode 路径的实际调用模式**（inplace + ssm_state_indices + varlen）。

---

## 5. 未覆盖场景

### 5.1 真实存在但未覆盖的场景

| 缺失场景 | 严重程度 | 说明 |
|----------|---------|------|
| `fused_kda_gate` 精度测试 | **高** | 门控计算 `g1 = A * softplus(beta * g)`，在 `KimiDeltaAttention.forward` 中被调用（`kda.py:264`），直接影响 attention 输入质量 |
| 零初始状态 | **低** | vLLM 中新请求的 initial_state 对应行被置零，测试中用随机 h0 覆盖；零值是随机值的特例，精度问题大概率能被随机状态覆盖 |
| D != 128（其他头维度） | **低** | 取决于模型配置，如果模型只使用 D=128 则无需覆盖 |

### 5.2 超出当前测试文件范畴的缺失

| 缺失场景 | 建议归属 |
|----------|---------|
| `FusedRMSNormGated` | 单独的 norm 测试文件 |
| `KimiDeltaAttention` 层级集成测试 | 模型层集成测试 |
| Prefill → Decode 状态传递 | 端到端集成测试 |
| `causal_conv1d_fn` / `causal_conv1d_update` | conv1d 测试文件 |

### 5.3 边界场景

| 缺失场景 | 严重程度 | 说明 |
|----------|---------|------|
| Prefill 路径极短序列 (T=1) | **中** | `test_chunk_kda_npu.py` 最小 T=15，T=1 的 chunk 边界未覆盖 |
| Prefill 路径极长序列 (T>8192) | **低** | 当前最大 8192，超长序列的内存/精度行为未知 |
| Decode 路径大批量 (N>16) | **低** | Test 2 最大 N=16，更大批量未覆盖 |

---

## 6. 总结

| 模块 | 测试文件 | 测试状态 | 评估 |
|------|---------|---------|------|
| `chunk_kda` (prefill) | `test_chunk_kda_npu.py` | **充分覆盖** | 参数取值与 vLLM 实际调用完全一致，9 组参数覆盖多种序列长度和模式 |
| `fused_recurrent_kda` (decode) | `test_fused_recurrent_kda_npu.py` | **充分覆盖** | 3 个测试函数 19 组参数，覆盖非原地/原地/float32 三种模式，Test 2 完全匹配 vLLM decode 调用模式 |
| `fused_kda_gate` | — | **未覆盖** | 在 KDA forward 中被调用，应补充测试 |
| 子算子 (cumsum, solve_tril, etc.) | — | 间接覆盖 | 通过 chunk_kda 端到端间接覆盖 |

### 建议补充的测试

1. **`fused_kda_gate` 精度测试** — 门控计算直接影响 attention 输入，优先级最高
2. **Prefill 极短序列 (T=1)** — 验证 chunk 边界正确性
3. **零初始状态测试** — `h0 = torch.zeros(...)` 模拟新请求首次 prefill