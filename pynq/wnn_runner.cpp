#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <time.h>

// -------------------------------------------------------------------------
// CONSTANTS
// -------------------------------------------------------------------------
#define REG_CTRL        0x00
#define REG_OUTPUT      0x04
#define CTRL_START_BIT  0
#define CTRL_DONE_BIT   1
#define CTRL_CLEAR_BIT  2

#define MM2S_DMACR      0x00 
#define MM2S_DMASR      0x04 
#define MM2S_SA         0x18 
#define MM2S_LENGTH     0x28 
#define DMASR_IDLE      (1 << 1)
#define DMA_BYTES       (784 * 4) 

void* map_phys_addr(off_t phys_addr, size_t len) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd == -1) exit(1);
    size_t pagesize = sysconf(_SC_PAGE_SIZE);
    off_t page_base = (phys_addr / pagesize) * pagesize;
    off_t page_offset = phys_addr - page_base;
    void* mapped_base = mmap(NULL, len + page_offset, PROT_READ | PROT_WRITE, MAP_SHARED, fd, page_base);
    close(fd);
    return (char*)mapped_base + page_offset;
}

int main(int argc, char* argv[]) {
    if (argc < 6) return -1;

    uint32_t wnn_phys  = strtoul(argv[1], NULL, 0);
    uint32_t dma_phys  = strtoul(argv[2], NULL, 0);
    uint32_t data_phys = strtoul(argv[3], NULL, 0);
    int      count     = atoi(argv[4]);
    char* gt_file      = argv[5];

    volatile uint32_t* wnn_regs = (volatile uint32_t*)map_phys_addr(wnn_phys, 0x1000);
    volatile uint32_t* dma_regs = (volatile uint32_t*)map_phys_addr(dma_phys, 0x1000);

    // Pre-calculate Pointers for Speed
    volatile uint32_t* dma_sa   = &dma_regs[MM2S_SA >> 2];
    volatile uint32_t* dma_len  = &dma_regs[MM2S_LENGTH >> 2];
    volatile uint32_t* dma_stat = &dma_regs[MM2S_DMASR >> 2];
    volatile uint32_t* wnn_ctrl = &wnn_regs[REG_CTRL >> 2];
    volatile uint32_t* wnn_out  = &wnn_regs[REG_OUTPUT >> 2];

    uint8_t* y_test = (uint8_t*)malloc(count);
    FILE* f = fopen(gt_file, "rb");
    if(fread(y_test, 1, count, f) != count) {} 
    fclose(f);

    // Reset DMA
    dma_regs[MM2S_DMACR >> 2] = 4; 
    while(dma_regs[MM2S_DMACR >> 2] & 4);
    dma_regs[MM2S_DMACR >> 2] = 1; 

    // Initial Cleanup
    *wnn_ctrl = (1 << CTRL_CLEAR_BIT);

    printf("[C++] Starting Loop...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int correct = 0;

    for (int i = 0; i < count; i++) {
        // 1. DMA Transfer
        *dma_sa  = data_phys + (i * DMA_BYTES);
        *dma_len = DMA_BYTES;

        // 2. Wait for DMA (Must happen BEFORE start)
        while( (*dma_stat & DMASR_IDLE) == 0 );

        // 3. Start Core
        *wnn_ctrl = (1 << CTRL_START_BIT);
        
        // 4. Clear Status & Stop Pulse
        *wnn_ctrl = (1 << CTRL_CLEAR_BIT);

        // 5. Wait for Done
        while( (*wnn_ctrl & (1 << CTRL_DONE_BIT)) == 0 );

        // 6. Read Result
        if (*wnn_out == y_test[i]) correct++;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;

    printf("\n[C++] FINAL REPORT:\n");
    printf("      Accuracy: %.2f%% (%d/%d)\n", (float)correct/count * 100.0f, correct, count);
    printf("      Time:     %.4f s\n", time_taken);
    printf("      FPS:      %.2f\n", count / time_taken);

    free(y_test);
    return 0;
}