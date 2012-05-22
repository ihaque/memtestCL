/*
 * xxd.cpp
 * Limited implementation of the Unix xxd utility to assist in
 * the MemtestCL build process under Windows.
 *
 * Author: Imran Haque, 2010
 * Copyright 2010, Stanford University
 *
 * This file is licensed under the terms of the LGPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

#include <stdio.h>
#include <string.h>

int main(int argc,char** argv) {
    if (argc < 2 || strcmp(argv[1],"-i")) {
            fprintf(stderr,"xxd -i [filename]\n");
            return 1;
    }
    FILE* f = fopen(argv[2],"rb");
    unsigned char block[16384];
    size_t nvalid = fread(block,1,16384,f);
    printf("unsigned char %s[] = {\n  ",argv[2]);
    int curcol = 0;
    unsigned totalbytes = 0;
    while (nvalid > 0) {
        for (int i = 0; i < nvalid;i++) {
            int j = block[i];
            printf("0x%02x, ",j);
            curcol++;
            totalbytes++;
            if (curcol == 12) {
                printf("\n  ");
                curcol = 0;
            }
        }
        nvalid = fread(block,1,16384,f);
    }
    printf("\n};\n");
    printf("unsigned int %s_len = %u;\n",argv[2],totalbytes);
    fclose(f);
    return 0;
}
