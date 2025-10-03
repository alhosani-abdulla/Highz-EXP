#include "LORegisterLibrary.h"

int GCD(int a, int b) {
    while (1) {
        if (a == 0){
            return b;
        }
        else if (b == 0){
            return a;
        }
        else if (a > b){
            a = a % b;
        }
        else{
            b = b % a;
        }
    }
}

double roundNearestEven(double x) {
    double intPart;
    double frac = modf(x, &intPart);

    if (fabs(frac) == 0.5) {
        if (fmod(intPart, 2.0) == 0.0) {
            return intPart;
        } else {
            return intPart + (x > 0 ? 1.0 : -1.0);
        }
    } else {
        return round(x);
    }
}

int findMin(int a, int b) {
    return (a < b) ? a : b;
}

void calcRegs(double freq, int *INT, int *MOD, int *FRAC, int *outputDiv, double *bandSelectClockDivider){
    int deviceType = 1;         //1 corresponds to ADF4351, 0 for ADF4350
    double refFreq = 25.0;
    int rCount = 1;
    bool refDouble = false;
    bool refDiv2 = false;
    int feedbackSelect = 1;    //1 corresponds to Fundamental, 0 for divider
    int bandSelectClockMode = 0;     //0 corresponds to Low, 1 for High
    bool GCDEnabled = true;

    double PDFFreq = refFreq / rCount;

    int tempOutputDivider;
    int i;
    for (i = 1; 1 < 7; i++){
        tempOutputDivider = pow(2, i);
        double quotient = 2200.0 / tempOutputDivider;
        if (quotient <= freq){
            break;
        }
    }
    //bug in code above; could not find; value hard coded to be 4 (correct)
    tempOutputDivider = 4;

    double N = freq * tempOutputDivider / PDFFreq;

    double tempINT = floor(N);
    tempINT = int(tempINT);

    double tempMOD = roundNearestEven(1000.0 * PDFFreq);
    tempMOD = int(tempMOD);

    double tempFRAC = roundNearestEven((N - tempINT) * tempMOD);
    tempFRAC = int(tempFRAC);

    if (GCDEnabled){
        int div = GCD(tempMOD, tempFRAC);
        tempMOD = tempMOD / div;
        tempFRAC = tempFRAC / div;
    }

    if (tempMOD == 1){
        tempMOD = 2;
    }

    //PDF Freq checks omitted for simplicity

    int PDFScale = 8;
    double tempBandSelectClockDivider = (double) findMin(ceil(PDFScale * PDFFreq), 255);
    double tempBandSelectClockFreq = 1000.0 * PDFFreq / tempBandSelectClockDivider;

    //bandSelectClockFreq checks omitted for simplicity

    *INT = tempINT;
    *MOD = tempMOD;
    *FRAC = tempFRAC;
    *outputDiv = tempOutputDivider;
    *bandSelectClockDivider = tempBandSelectClockDivider;
}

void makeRegs(int INT, int FRAC, int MOD, int outputDiv, int bandSelectClockDiv, uint32_t *reg0, uint32_t *reg1, uint32_t *reg2,
    uint32_t *reg3, uint32_t *reg4, uint32_t *reg5) {

    int rCounter = 1;

    //skipping check values to ensure no error (for simplicity)

    int outputDivSelect = (int) (log(outputDiv) / log(2));

    //skipping check that outputDivSelect is a power of 2

    uint32_t tempReg0 = (uint32_t)INT << 15 | (uint32_t)FRAC << 3 | 0x0;

    uint32_t tempReg1 = (0UL << 28) | (1UL << 27) | (1UL << 15) | ((uint32_t)MOD << 3) | 0x1UL;
    //since the device is the ADF4351, we also have the following to add...
    tempReg1 |= (0UL << 28);

    uint32_t tempReg2 = (0 << 29 | 0 << 26 | 0 << 25 | 0 << 24 | rCounter << 14 |
        0 << 13 | 7 << 9 | (FRAC == 0 ? 0 : 1)  << 8 | 0 << 7 | 1 << 6 |
        0 << 5 | 0 << 4 | 0 << 3 | 0x2);
    
    uint32_t tempReg3 = (0 << 18 | 0 << 15 | 150 << 3 | 0x3);
    //since the device is the ADF4351, we also have the following to add...
    tempReg3 |= ((0 << 23 | 0 << 22 | 0 << 22));

    uint32_t tempReg4 = (1UL << 23) | ((uint32_t)outputDivSelect << 20) | ((uint32_t)bandSelectClockDiv << 12) | (0UL << 11) |
        (0UL << 10) | (0UL << 9) | (0UL << 8) | (0UL << 6) | (1UL << 5) | (3UL << 3) | 0x4UL;
    
    uint32_t tempReg5 = (1UL << 22 | 3UL << 19 | 0x5UL);

    *reg0 = tempReg0;
    *reg1 = tempReg1;
    *reg2 = tempReg2;
    *reg3 = tempReg3;
    *reg4 = tempReg4;
    *reg5 = tempReg5;
}