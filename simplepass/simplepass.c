#include "hal.h"
#if HAL_TYPE == HAL_stm32f3
#include "stm32f303x8.h"
#include "core_cm4.h"
#endif
#include "simpleserial.h"
#include <stdint.h>
#include <stdlib.h>
#define CLK_NOW     DWT->CYCCNT
#define DOOR_OPEN   0xA5
#define DOOR_CLOSED 0x5A
#define SAFEBOOL_TRUE  0xA5A5A5A5A5A5A5A5ull
#define SAFEBOOL_FALSE 0x5A5A5A5A5A5A5A5Aull
const char secr[] = "verysafe";
const char pstr[] = "Pass";
const char fstr[] = "Fail";
volatile int door;
void validate(uint8_t* pass)
{
  door = DOOR_CLOSED;
  //volatile uint64_t good = SAFEBOOL_TRUE;
  volatile uint32_t good = SAFEBOOL_TRUE;
  for(int i=0;i<sizeof(secr);i++)
  {
    if(pass[i] != secr[i])
    {
      good = SAFEBOOL_FALSE;
    }
  }
  //if(good == (uint64_t)SAFEBOOL_TRUE)
  if(good == (uint32_t)SAFEBOOL_TRUE)
  {
    door = DOOR_OPEN;
  }
}
void cyccnt_init()
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  //ETM->LAR = 0xC5ACCE55;
  ITM->LAR = 0xC5ACCE55;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
uint8_t callback(uint8_t* pass, uint8_t len)
{
  trigger_high();
  validate(pass);
  trigger_low();
  simpleserial_put('r',sizeof(fstr),door==DOOR_OPEN?pstr:fstr);
  return 0;
}
int main(void)
{
  platform_init();
  cyccnt_init();
  init_uart();
  trigger_setup();
  simpleserial_init();
  simpleserial_addcmd('p', sizeof(secr), callback);
  while(1)
  {
    simpleserial_get();
  }
}
