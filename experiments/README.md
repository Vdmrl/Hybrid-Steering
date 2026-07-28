# Experiment contract

Каждый эксперимент выполняется в собственной ветке:

```text
exp/<concept>-<ablation>
```

Если среда требует служебный префикс, допустимо
`<prefix>/exp/<concept>-<ablation>`. Запускать новый эксперимент из `main`
нельзя.

В ветку входят только:

- точный config или manifest;
- небольшой воспроизводимый fixture;
- компактная итоговая сводка.

Полные генерации, веса, `.env` и API-ключи в Git не добавляются.

Все методы записывают zero-based абсолютные decoder-слои и совместимые
`GenerationRecord`. Благодаря этому GDN, activation steering, SVD и layer
ablation можно оценивать одним Judge.
