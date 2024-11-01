"""Code to give a rough estimation of the number of possible 'unique' x86 assembly lines"""

from x86_check_asm import X86_64_ISA, MEMORY_ACCESS_GP_REGISTERS
from bincfg.normalization.x86.x86_tokenizer import X86_REGISTER_SIZES

import re

num_imm = 3

def num_reg(size):
    return len([k for k, v in X86_REGISTER_SIZES.items() if v == int(size) // 8])

def num_bnd_reg():
    # These BND registers existed in intel MPX, but they have since been deprecated
    return 4

def num_cr_reg():
    return 16

def num_segment_reg():
    return 6

def num_mm(t):
    t = '' if t is None else t
    return len([k for k in X86_REGISTER_SIZES if k.startswith(t + 'mm')])

def num_bcst():
    # This will always be one because there will only ever be one way of broadcasting a 32-bit or 64-bit float to
    #   whatever the size of the current registers
    return 1

def num_st_reg():
    return len([k for k in X86_REGISTER_SIZES if k.startswith('st(')])

def num_rounding_modes(r):
    if r == '{er}':
        return 4
    elif r == '{sae}':
        return 2
    raise ValueError("Unknown rounding mode string: %s" % repr(r))

_CHECK_AVX_OPMASK_RE = re.compile(r'{k(?:[0-7])}({z})?')
num_avx_opmask_regs = 8
def num_avx_opmask(s):
    # There are 8 'k' registers. k0 can only be used in some cases, but this is an upper bound
    if _CHECK_AVX_OPMASK_RE.fullmatch(s) is None:
        raise ValueError("Unknown avx opmask: %s" % repr(s))
    groups = _CHECK_AVX_OPMASK_RE.fullmatch(s).groups()
    return num_avx_opmask_regs * (2 if groups[0] is not None else 1)  # check for zeroed or merged modes

num_mem_reg = len(MEMORY_ACCESS_GP_REGISTERS) + 1
num_scale = 4
def num_mem():
    # We can have [r], [i], [r + i], [r + r], [r*i], [r + r*i], [r*i + i], [r + r + i], [r + r*i + i],
    #   as well as different orderings of those. We don't have to worry about the orderings right now...
    # Since this is memory addressings, we can only use the MEMORY_ACCESS_GP_REGISTERS for registers, 1,2,4,or 8 for
    #   index scale, and sometimes the rip register. We will just make an upper bound for now and include the rip reg
    return num_mem_reg + num_imm + (num_mem_reg * num_imm) + (num_mem_reg * num_mem_reg) + (num_mem_reg * num_scale) +\
        (num_mem_reg * num_mem_reg * num_scale) + (num_mem_reg * num_scale * num_imm) + (num_mem_reg * num_mem_reg * num_imm) +\
        (num_mem_reg * num_mem_reg * num_scale * num_imm)

AVX_OPMASK_RE = r'({k[0-7]}(?:{z})?)'
AVX_REG_RE = r'([xyz]?)mm[0-4]?'
AVX_MEM_RE = r'm(8|16|32|64|128|256|512)'
AVX_BCST_RE = r'm(32|64)bcst'
AVX_ROUNDING_RE = r'({er}|{sae})'
AVX_RE = re.compile(r'{reg} ?{opmask}?(?:/{mem})?(?:/{bcst})? ?{rnd}?{opmask}?'\
    .format(reg=AVX_REG_RE, opmask=AVX_OPMASK_RE, mem=AVX_MEM_RE, bcst=AVX_BCST_RE, rnd=AVX_ROUNDING_RE))

REG_MEM_RE = re.compile(r'(reg|r|m|r(?:8|16|32|64|128|256|512)?/m|reg/m)(8|16|32|64|128|256|512)[abcd]?(?:fp|int)? ?({round})?'.format(round=AVX_ROUNDING_RE))

# Captures in order:
#   0. avx_reg_type ('x', 'y', or 'z')
#   1. opmask_str ('{k[0-7]}{z}')
#   2. memsize (32, 64, 128, 256, 512)
#   3. broadcast_memsize (32, 64)
#   4. mem_rounding ('{er}')
#   5. final_opmask_str ('{k[0-7]}{z}')

# existed in intel MPX, but has since been deprecated
BND_RE = re.compile(r'bnd[1-2]?(?:/m(64|128))?')

def mult_type(spec):
    ret = 1
    for v in spec:
        if re.fullmatch(r'al|[er]?ax|<xmm0>|rdi|rsi|0|1|dx|cr8|[defgsc]s|cl', v) is not None:
            c = 1
        elif re.fullmatch(r'imm(?:8|16|32|64)|rel(?:8|16|32)|ptr16:16|ptr16:32|disp16/32', v) is not None:
            c = num_imm
        elif REG_MEM_RE.fullmatch(v) is not None:
            groups = REG_MEM_RE.fullmatch(v).groups()
            c = (num_reg(groups[1]) if '/m' in groups[0] else 0) + (num_mem() if '/m' in groups[0] else 0)
            c *= (num_rounding_modes(groups[2]) if groups[2] is not None else 1)
        elif AVX_RE.fullmatch(v) is not None:
            groups = AVX_RE.fullmatch(v).groups()
            # Matching in order: register, opmask, memory, broadcasted memory
            # We multiply the register and opmask + 1, then add the memory, then add broadcasting (memory * num_bcst)
            c = num_mm(groups[0]) * (1 + (num_avx_opmask(groups[1]) if groups[1] is not None else 0)) + \
                (num_mem() if groups[2] is not None else 0) + \
                (num_mem() * (num_bcst() if groups[3] is not None else 0))
            
            # Multiply by number of rounding modes if present
            c *= num_rounding_modes(groups[4]) if groups[4] is not None else 1
            
            # Multiply by final opmask if using
            c *= num_avx_opmask(groups[5]) if groups[5] is not None else 1
        elif re.fullmatch(BND_RE, v) is not None:
            groups = BND_RE.fullmatch(v)
            c = num_bnd_reg() + (num_mem() if groups[1] is not None else 0)
        elif re.fullmatch(r'mib|m16&16|m32&32|m16&32|m16&64', v) is not None:
            # This is another one of those BND instruction things. I think it is a memory address?
            c = num_mem()
        elif re.fullmatch(r'm16:16|m16:32|m16:64', v) is not None:
            # This happens with call far instructions, I think it's memory address?
            c = num_mem()
        elif re.fullmatch(r'm80dec|m80bcd|m80fp|m2byte|m512byte', v) is not None:
            # This happens with some FP instructions, it's a memory address
            c = num_mem()
        elif re.fullmatch(r'moffs(?:8|16|32|64)', v) is not None:
            # Something for segmented move instructions
            c = num_mem()
        elif re.fullmatch(r'm(?:256)?|mem|vm(?:32|64)[xyz]', v) is not None:
            # General memory location
            c = num_mem()
        elif re.fullmatch(r'm14/28byte|m94/108byte', v) is not None:
            # Another FP instruction, can be either memory or immediate
            c = num_mem() + num_imm
        elif re.fullmatch(r'k[1-2] {k[1-2]}', v) is not None:
            # Specifically just using the opmask registers as registers
            c = num_avx_opmask_regs * num_avx_opmask_regs
        elif re.fullmatch(r'k[1-3]', v) is not None:
            # Specifically just using the opmask registers as registers, no extra opmask
            c = num_avx_opmask_regs
        elif re.fullmatch(r'k[1-3]/m(?:8|16|32|64)', v) is not None:
            # Specifically just using the opmask registers as registers, no extra opmask, with memory
            c = num_avx_opmask_regs + num_mem()
        elif re.fullmatch(r'reg', v) is not None:
            # Any normally sized register
            c = num_reg(8) + num_reg(16) + num_reg(32) + num_reg(64)
        elif re.fullmatch(r'st(?:\((?:[0-8]|i)\))?', v) is not None:
            c = num_st_reg()
        elif re.fullmatch(r'cr0-cr7|dr0-dr7', v) is not None:
            # Specifically the first 8 cr or dr registers
            c = 8
        elif re.fullmatch(r'sreg', v) is not None:
            # The number of segment registers
            c = num_segment_reg()
        elif re.fullmatch(r'(?:m(?:32|64)|vm(?:32|64)[xyz]) {k1}', v) is not None:
            c = num_mem() * num_avx_opmask('{k1}')
        elif re.fullmatch(r'any', v) is not None:
            # This is something my code does for nop instructions
            c = 1
        else:
            raise NotImplementedError(v)
        ret *= c
    return ret

def get_count():
    total_count = 0
    for k, specs in X86_64_ISA.items():
        for spec in specs:
            try:
                total_count += mult_type(spec)
            except:
                print("Failed on", k, specs)
                raise
    return total_count


if __name__ == '__main__':
    print(get_count())