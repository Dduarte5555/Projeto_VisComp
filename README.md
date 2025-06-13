# mri-cv-tools

Biblioteca Python para processamento de imagens médicas:
1. **Treinamento de modelos com diferentes dimensionalidades**  
2. **Inferência dos modelos em exames para a criação e visualização de máscaras**  
3. **Tirada de métricas do modelo para a validação dele**  

## Instalação

```bash
git clone https://github.com/Dduarte5555/Projeto_VisComp
cd Projeto_VisComp
pip install .
```

**É recomendado que se use um environment com python 3.11.13 instalado. O programa não foi testado utilizando outras versões de python, e pode apresentar problemas inesperados**

Caso seja necessário, você pode instalar um env de conda com a versão de python apropriada utilizando o comando

```conda create --name myenv python=3.11```


## Links relevantes
[Versão original do dataset usado para o treinamento do modelo](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101942541)

[Arquitetura usada para a segmentação 2D e 2.5D](https://arxiv.org/abs/1807.10165)

[Código base para o treinamento do modelo 3D de segmentação](https://github.com/bnsreenu/python_for_microscopists/blob/master/215_3D_Unet.ipynb)

# Scripts de validação
Para a validação do programa, uma série de scipts foram montados, que demonstram a usabilidade da biblioteca, e como ela funciona.

```python test_inference.py```
```python test_metrics.py.py```
```python test_train.py.py```



## Funções de treinamento

train3d.train_model3d, train2_5d.train_model2d e train2_5d.train_model2_5d  
São usadas para fazer o treinamento individual de cada modelo.
As funções usam os seguintes argumentos:  

train_model3d
    path - Caminho para o dataset 
    filename - nome do arquivo do modelo quando treinado,
    batch_size, epochs, patch_size, n_classes, channels, LR - hiperparâmetros para o modelo, todos possuem um valor padrão (que também foram os valores usados para o treinmaento dos modelos) para facilitar o treinamento  

train_model2d
    path - Caminho para o dataset 
    filename - nome do arquivo do modelo quando treinado,
    LR, N_EPOCHS, BATCH_SIZE, THRESH_IoU , classes - hiperparâmetros para o modelo, todos possuem um valor padrão (que também foram os valores usados para o treinmaento dos modelos) para facilitar o treinamento 

train_model2_5d
    path - Caminho para o dataset 
    filename - nome do arquivo do modelo quando treinado,
    win_size - tamanhgo da janela que será usada para o treinamento do modelo 2.5D
    LR, N_EPOCHS, BATCH_SIZE, THRESH_IoU , classes - hiperparâmetros para o modelo, todos possuem um valor padrão (que também foram os valores usados para o treinmaento dos modelos) para facilitar o treinamento

A função train.train_all faz o treinamento dos 3 modelos, sequencialmente

train_all
    dataset_path- Caminho para o dataset 
    batch_size, epochs, patch_size, n_classes, THRESH_IoU_2D, THRESH_IoU_2_5D, LR_2D, LR_2_5D, LR_3D - Hiperparâmetros distribuidos para cada função de treinamento.

A função train_test.test_training faz um treinamento limitado em 5 epochs para o dataset especificado. Tem como intuito ser mais um sanity check, para garantir que o treinamento dos 3 modelos ocorrera de forma adequada, quando usando uma maior quantidade de epochs, e um dataset maior.

test_training
    path - Caminho para o dataset de teste

## Funções para inferência e visualização

Para fazer a inferência de um modelo, existem 3 funções que cumprem esse propósito

Inference.load_and_run_inference_2D, Inference.load_and_run_inference_2_5D e Inference3d.run_3d_inference

load_and_run_inference_2_5D - 
    image_path - path para a imagem
    model_path - path para o modelo
    window_size - tamanho da jkanela

load_and_run_inference_2D- 
    image_path - path para a imagem
    model_path - path para o modelo

run_3d_inference-
    image_path - path para a imagem
    model_path - path para o modelo

## Testando a eficácia dos modelos

Para testar a eficácia de um modelo, a função Inference.evaluate_folder faz a inferencia de um diretório com imagens de validação, procura por máscaras que tenham um match para fazer a validação, e tira as métricas apropriadas.

evaluate_folder(
    images_dir="diretorio com as imagens de validacao", 
    model_path="caminho para o modelo", 
    inference_function=funcao_de_inferencia
)