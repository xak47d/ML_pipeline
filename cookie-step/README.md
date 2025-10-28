# Plantilla Cookiecutter para pasos de MLFlow

Con esta plantilla puedes generar rápidamente nuevos pasos para ser usados con MLFlow.

# Uso
Ejecuta el comando:

```
> cookiecutter [ruta a este folder] -o [directorio de destino]
```

y sigue las indicaciones. La herramienta te pedirá el nombre del paso, el nombre del script, la descripción, etc. También te pedirá los parámetros del script. Esto debe ser una lista de nombres de parámetros separados por comas *sin espacios*. 
Cuando termines, encontrarás un nuevo directorio con el nombre del paso proporcionado que contiene una plantilla básica de un nuevo paso de MLflow.

Necesitarás editar tanto el script como los archivos MLproject para completar el tipo y la ayuda de los parámetros.
Por supuesto, si tu script necesita paquetes, estos deben agregarse al archivo conda.yml.
