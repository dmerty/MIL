# Распределение пикселей по классам:
![Image alt](https://s1.hostingkartinok.com/uploads/images/2022/11/3da0301de49d2fa6d0e00fb22f86a85e.png)

# Примеры исходных изображений с аугментацией:
![Image alt](https://s1.hostingkartinok.com/uploads/images/2022/11/478a5998f629c6fa83585aa7d6f65f60.png)

# Примеры изображений с предсказаниями модели:
![Image alt](https://s1.hostingkartinok.com/uploads/images/2022/11/d877f46d957f77f0d399672ab69e3260.png)

# Метрики качества модели:
metric |body 
---    |---           
iou0   | 0.732

metric |upper_body |lower_body
---    |---        |---      
iou1   | 0.74      | 0.396

metric |low_hand |torso |low_leg |head  |up_leg |up_hand 
---    |---      |---   |---     |---   |---    |---         
iou2   | 0.381   |0.509 |0.278   |0.777 |0.32   |0.381

# Выводы:
Кажется, что модель неплохо справляется с поставленной задачей, для улучшения качества можно попробовать альтернативные подходы к обучению (архитектуру, функцию потерь, оптимизатор и пр.)
