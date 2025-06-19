# Progetto SCPA - Calcolo parallelo per il prodotto tra una matrice sparsa e un vettore 

L’obiettivo di questo lavoro è quello di sviluppare un nucleo di calcolo per il prodotto tra una matrice sparsa ed un vettore, che sia in grado di calcolare:

y ← Ax

La matrice è memorizzata nei formati:

• CSR

• HLL

Il nucleo è stato parallelizzato tramite OpenMP e CUDA ed è stato collaudato confrontando i risultati paralleli con quelli ottenuti da un’implementazione seriale.

Passi per l'esecuzione:

## Esecuzione tramite OpenMP
Dalla cartella del progetto eseguire i seguenti comandi: 

1. Test al variare dei thread
   
        ./openmp_threads.sh
   
2. Test con numero di thread fissato, configurabile nel file openmp.sh
   
        ./openmp.sh

## Esecuzione tramite CUDA
Dalla cartella del progetto eseguire i seguenti comandi: 

1. Test al variare della dimensione del blocco con tutte le versioni del prodotto CUDA (CSR thread per riga, CSR warp per riga, HLL thread per riga, HLL warp per riga)
   
        ./cuda_test.sh
   
2. Test con la migliore configurazione ottenuta (threads_per_block = 128, CSR warp per riga, HLL thread per riga)
   
        ./cuda.sh
