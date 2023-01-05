## Exercise 1

```shell
richwill@soctf-pdc-001:~$ lscpu
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   46 bits physical, 48 bits virtual
CPU(s):                          20
On-line CPU(s) list:             0-19
Thread(s) per core:              2
Core(s) per socket:              10
Socket(s):                       1
NUMA node(s):                    1
Vendor ID:                       GenuineIntel
CPU family:                      6
Model:                           85
Model name:                      Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz
Stepping:                        4
CPU MHz:                         801.289
CPU max MHz:                     3000.0000
CPU min MHz:                     800.0000
BogoMIPS:                        4400.00
Virtualization:                  VT-x
L1d cache:                       320 KiB
L1i cache:                       320 KiB
L2 cache:                        10 MiB
L3 cache:                        13.8 MiB
NUMA node0 CPU(s):               0-19
Vulnerability Itlb multihit:     KVM: Mitigation: Split huge pages
Vulnerability L1tf:              Mitigation; PTE Inversion; VMX conditional cache flushes, SMT vulnerable
Vulnerability Mds:               Mitigation; Clear CPU buffers; SMT vulnerable
Vulnerability Meltdown:          Mitigation; PTI
Vulnerability Mmio stale data:   Mitigation; Clear CPU buffers; SMT vulnerable
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Retpolines, IBPB conditional, IBRS_FW, STIBP conditional, RSB filling
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Mitigation; Clear CPU buffers; SMT vulnerable
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe sy
                                 scall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni
                                  pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt
                                 tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single pti in
                                 tel_ppin ssbd mba ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2
                                  erms invpcid rtm cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsave
                                 opt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts hwp hwp_act_window hwp
                                 _epp hwp_pkg_req pku ospke md_clear flush_l1d arch_capabilities
```

What is a socket and how many do you have?

- A socket is a receptacle on the motherboard for one physically packaged processor (each of which can contain one or more cores).

What are, and what are the relationships between CPUs, cores, and threads?

- The CPUs from `lscpu` are the logical CPU
- In general, Cores * Threads per core = l-Logical CPU
- More accurately, Sockets * Cores per socket * Threads per core = Logical CPU

What are the different levels of cache present and how large are they?

- L1d cache: 320 KiB
- L1i cache: 320 KiB
- L2 cache: 10 MiB
- L3 cache: 13.8 MiB

## Exercise 2

`sinfo` is used to view information about Slurm nodes and partitions.

```shell
richwill@soctf-pdc-001:~$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
global*      up    3:00:00     10   idle soctf-pdc-[006,008,013-016,019,021,023-024]
global*      up    3:00:00      2   down soctf-pdc-[005,007]
dxs-4114     up    3:00:00      1   idle soctf-pdc-019
i7-7700      up    3:00:00      4   idle soctf-pdc-[013-016]
i7-9700      up    3:00:00      1   idle soctf-pdc-021
xs-4114      up    3:00:00      2   idle soctf-pdc-[006,008]
xs-4114      up    3:00:00      2   down soctf-pdc-[005,007]
xw-2245      up    3:00:00      2   idle soctf-pdc-[023-024]
```

## Exercise 3

`srun` is used to run parallel jobs.

## Exercise 4

`sacct` is used to displays accounting data for all jobs and job steps in the Slurm job accounting log or Slurm database.

## Exercise 11
