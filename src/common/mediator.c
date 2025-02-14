#include "../../lib/utils.h"

MatrixConversionMediator createMatrixMediator() {
    MatrixConversionMediator mediator;
    mediator.convertToCSR = convertCOOtoCSR;
    mediator.convertToHLL = convertCOOtoHLL;
    return mediator;
}

