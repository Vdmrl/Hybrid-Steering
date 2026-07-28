# Hybrid Steering Core

Минимальные общие операции над recurrent state гибридных LLM:

- найти GDN-слои;
- извлечь recurrent state;
- вычислить среднее направление `positive - negative`;
- прибавить направление или пересадить состояние;
- сохранить направление вместе с воспроизводимыми метаданными.

## Установка

```bash
pip install -e "steering[dev]"
```

## Пример

```python
from hybrid_steering import (
    add_direction,
    extract_recurrent,
    mean_direction,
    subtract_states,
)

differences = [
    subtract_states(positive_state, negative_state)
    for positive_state, negative_state in paired_states
]
direction = mean_direction(differences)
add_direction(cache, direction, alpha=4.0, layers=[0, 1, 2])
```

`layers` — всегда абсолютные zero-based индексы decoder-слоёв. Это не
порядковые номера среди GDN-слоёв: первый GDN-слой Qwen3.5 имеет индекс `0`.

Перед и после вмешательства можно проверить, что KV и convolution state не
изменились:

```python
before = snapshot_nonrecurrent(cache)
add_direction(cache, direction, alpha=4.0)
assert_nonrecurrent_unchanged(before, cache)
```

SVD, clamp, gated injection и конкретные experiment runners намеренно не входят
в общее ядро. Сначала они проверяются в отдельных `exp/...` ветках.
