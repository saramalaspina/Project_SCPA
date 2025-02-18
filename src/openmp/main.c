#include <stdio.h>
#include <stdlib.h>
#include "../../lib/utils.h"

int main(int argc, char *argv[]) {

    if (argc < 3) {
        fprintf(stderr, "Usage: %s [matrix-market-filename] [format]\n", argv[0]);
        exit(1);
    }    

    MatrixElement *mat = read_matrix(argv[1]);
    if (!mat) exit(1);

    // Creazione del mediatore
    MatrixConversionMediator mediator = createMatrixMediator();

    char *e = argv[2];

    switch(e[0]){
        case 'S': 
            serialExecutionOpenmp(mat, mediator);
            break;
        case 'H':
            hllExecutionOpenmp(mat, mediator);
            break;
        case 'C':
            csrExecutionOpenmp(mat, mediator);
            break;
        default:
            printf("Tipo di esecuzione non riconosciuto\n");    
    }

    return 0;
}
