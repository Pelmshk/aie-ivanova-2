# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- Для части A выбран датасет STL10 (по рекомендации).
- Для части B выбран датасет OxfordIIITPet с треком `segmentation` для того, чтобы лучше изучить процесс сегментации.
- В части A сравнение моделей происходило по best_val_accuracy, эффект CNN, аугментаций и transfer learning; в части B сравнение моделей происходило по mean_iou и визуализированным маскам, проверялось влияние выбранного порога для бинаризации выбранного foreground-класса

## 2. Среда и воспроизводимость

- Python: 3.14.0
- torch: 2.10.0+cpu
- torchvision: 0.25.0+cpu
- Устройство (CPU/GPU): CPU
- Seed: 19
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: `STL10`
- Разделение: train/val/test
- Базовые transforms:  basic_transform (ToTensor, Normalize)
- Augmentation transforms: train_transform (RandomCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize), resnet_transform (Resize, ToTensor, Normalize)
- STL10 содержит 10 классов (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck) с изображениями 96×96 RGB. Задача сложная из-за малого числа размеченных примеров и разнообразия поз/освещения, что требует сильных аугментаций и transfer learning с ImageNet.

### 3.2. Часть B: structured vision

- Датасет: `OxfordIIITPet`
- Трек: `segmentation`
- Что считается ground truth: trimap маски из annotations/trimaps/ (*.png), где пиксели: 1=background, 2=pet (foreground), 3=border. GT = бинарная маска (2=pet → 1, остальное → 0).
- Какие предсказания использовались: DeepLabV3_ResNet50 с весами COCO_WITH_VOC_LABELS_V1
- OxfordIIITPet идеален для pet segmentation: 37 пород кошек/собак (~7400 изображений) с точными trimap масками на пиксельном уровне. Задача фокусируется на выделении животных на сложных фонах, что проверяет качество предобученных COCO весов на transfer к новым классам. Разумно использовать вероятностный threshold вместо жесткого argmax для повышения IoU.

## 4. Часть A: модели и обучение (C1-C4)

Опишите коротко и сопоставимо:

- C1 (simple-cnn-base): SimpleCNN 3conv+2fc, простая CNN без аугментаций
- C2 (simple-cnn-aug): SimpleCNN 3conv+2fc, C1 же CNN, но с разумными аугментациями
- C3 (resnet18-head-only): ResNet18 pretrained, head-only, ResNet18 с pretrained weights; backbone заморожен, обучается только классификационная голова.
- C4 (resnet18-finetune): ResNet18 pretrained, layer4+fc, ResNet18 с pretrained weights; частичное fine-tune, layer4 + fc

Дополнительно:

- Loss: CrossEntropyLoss()
- Optimizer(ы): C1/C2: Adam(lr=0.001), C3: Adam(fc params, lr=0.001), C4: Adam(layer4: lr=1e-4, fc: lr=1e-3, weight_decay=1e-4)
- Batch size: 64
- Epochs (макс): 20
- Критерий выбора лучшей модели: best_val_accuracy

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран segmentation track

- Модель: DeepLabV3_ResNet50 
- Что считается foreground: PET_IDX = min(CAT_IDX, DOG_IDX) из weights.meta["categories"]
- V1: базовая постобработка argmax(21 классов) == PET_IDX (жесткий выбор класса с максимальным логитом)
- V2: альтернативная постобработка softmax(PET_IDX) > 0.1 (изменен threshold с 0.5 на 0.1)
- Как считался mean IoU: По подмножествe изображений датасета: IoU = intersection/union для бинарных масок (pred vs GT), усреднено по изображениям
- Считались ли дополнительные pixel-level метрики: нет

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `./artifacts/figures/segmentation_examples.png`, `./artifacts/figures/segmentation_metrics.png`

Короткая сводка (6-10 строк):

- Лучший эксперимент части A: C4
- Лучшая `val_accuracy`: 0.945
- Итоговая `test_accuracy` лучшего классификатора: 0.943125
- Что дали аугментации (C2 vs C1): повышение accuracy (RandomCrop+Flip+ColorJitter сильно помогают на малом STL10) 
- Что дал transfer learning (C3/C4 vs C1/C2): сильное повышение accuracy (~0.94 по сравнению ~0.65)
- Что оказалось лучше: head-only или partial fine-tuning: partial fine-tuning немного выше по val_accuracy
- Что показал режим V1 во второй части: жесткий, много пропусков
- Что показал режим V2 во второй части: низкий threshold повышает качество модели
- Как интерпретируются метрики второй части: mean_iou показывает, что V2 лучше V1 (mean_iou V2 выше, чем mean_iou V1)

## 7. Анализ

(8-15 предложений)

- Простая CNN (C1/C2) на STL10 показывает низкую точность из-за малого объема данных и сложности задачи - 10 классов с разнообразными позами.
- Аугментации (C2 vs C1) дали устойчивое улучшение, стабильно работая через RandomCrop/Flip/ColorJitter, что компенсирует недостаток данных.
- Pretrained ResNet18 (C3/C4) взлетела до 94.5% благодаря ImageNet весам - модель "знает" низкоуровневые признаки (края, текстуры), которые универсальны для STL10.
- Head-only (C3) быстро достигла высокой accuracy, но partial fine-tuning (C4) добавил accuracy за счет адаптации layer4 под специфичные STL10 паттерны (дифференциальный lr: layer4=1e-4, fc=1e-3).
- Mean IoU идеальна для pet segmentation: напрямую измеряет пиксельное совпадение предсказанной и истинной маски, учитывая как полноту, так и точность локализации животного.
- Низкий threshold=0.1 вместо argmax позволил захватить контуры, хвосты, уши - именно там V1 "обрезал" слабые сигналы PET класса.
- Наиболее показательные ошибки V1: пропуски краевых пикселей питомцев. V2 устраняет систематический False Negative bias, жертвуя precision ради полноты - правильный tradeoff для pet detection на сложных фонах.

## 8. Итоговый вывод

(3-7 предложений)

- Базовый конфиг классификации: C4 (ResNet18 partial fine-tune layer4+fc) - дает повышение accuracy при стабильной обобщаемости. Комбинация ImageNet весов + тонкая настройка последних слоев под STL10 оказалась оптимальной по точность/вычислительные затраты.
- Что главное вы поняли про transfer learning: Pretrained веса решают проблемы на малых датасетах. ImageNet дает универсальные низкоуровневые признаки (края, градиенты), а fine-tuning верхних слоев адаптирует семантику под задачу. Head-only обучается быстро, но partial FT дает большую accuracy за счет доменной адаптации.
- В задачах сегментаIoU показывает, насколько точно модель рисует контуры питомцев. V1 (argmax) пропускала края и хвосты. V2 с threshold=0.1 нарисовала более полные силуэты.