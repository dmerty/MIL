# Распределение пикселей по классам:
![Image alt](https://s1.hostingkartinok.com/uploads/images/2022/11/3da0301de49d2fa6d0e00fb22f86a85e.png)

# Примеры исходных изображений с аугментацией:
![Image alt](https://s1.hostingkartinok.com/uploads/images/2022/11/478a5998f629c6fa83585aa7d6f65f60.png)

# Примеры изображений с предсказаниями модели:
![Image alt](https://s1.hostingkartinok.com/uploads/images/2022/11/57280a0842c86e3daffb1aa70ef6fee4.png)

# Значение функции потерь во время обучения:
![Image alt](https://s1.hostingkartinok.com/uploads/images/2022/11/17e3cf9207f0d46e617f9ee5ec1c7e6d.png)

# Значение ключевой метрики во время обучения:
![Image alt](https://s1.hostingkartinok.com/uploads/images/2022/11/9f36bbbeb2ec94cb204a2b00d74cce15.png)

# Метрики качества лучшей модели:
metric |body 
---    |---           
iou0   | 0.775

metric |upper_body |lower_body
---    |---        |---      
iou1   | 0.776     | 0.467

metric |low_hand |torso |low_leg |head  |up_leg |up_hand 
---    |---      |---   |---     |---   |---    |---         
iou2   | 0.43    |0.577 |0.354   |0.82  |0.367  |0.44

# Выводы:
Кажется, что модель неплохо справляется с поставленной задачей, для улучшения качества можно попробовать альтернативные подходы к обучению (архитектуру, функцию потерь, оптимизатор и пр.)
