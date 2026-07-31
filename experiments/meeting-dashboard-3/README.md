# Meeting dashboard 3

Русскоязычный аналитический слой для Experiment #3 (normalized feature
composition). Он собирается из компактных, уже закоммиченных сводок и не
требует API или GPU:

```powershell
python experiments/meeting-dashboard-3/build.py
```

По умолчанию результат записывается в
`outputs/meeting-dashboard-3/index.html`. Dashboard объединяет:

- результаты Exp3: rank-1/rank-4 SVD, post-sum truncation, RSS-нормализацию,
  классический activation steering, holdout и joy+optimism;
- таблицу всех 15 непустых комбинаций rank-1 GDN с observed joint rate,
  независимым baseline и разницей между ними;
- ключевые результаты строгого A/B factorial-теста из Dashboard 1;
- русские выводы, ограничения и раздел о том, какие эффекты уже можно
  считать убедительными.

Это аналитический отчёт, а не новый эксперимент: генерации и Judge API не
запускаются.
