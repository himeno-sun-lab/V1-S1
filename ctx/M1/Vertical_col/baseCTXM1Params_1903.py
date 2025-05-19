# This is the parameter file for the M1 motor cortex region

#######################################################################################################################
# This part is showing the the values of connection matrix including information of "connection probability", " sigma",
# "connection weight" and "synaptic delay". These for items are simply called :
# Probability: P
# Sigma:       S
# Weight:      W
# Delay:       D
# Excitatory   e
# Inhibitory   i
# L1           O
# L23          1
# L5A          2
# L5B          3
# L6           4
# Cell types:
# Excitatory cells: 'L23CC', 'L5ACS', 'L5ACC', 'L5ACT', 'L5BCS', 'L5BCC', 'L5BPT', 'L6CT',    total: 8
# Inhibitort cells: 'L1SBC', 'L1ENGC', 'L23PV', 'L23SST','L23VIP', 'L5APV', 'L5ASST',  'L5BPV', 'L5BSST', 'L6PV', 'L6SST',     total:11
# CC           Corticocortical
# CS           Corticospinal
# CT           Corticothalamic
# PT           Pyramidal tract
# SBC
# ENGC
# PV
# SST
# VIP
#
# the general form of parameters include the information of
# 1-main parameter, 2-postsynaptic cell 3-presynaptic cell, 4-target layer, 5-source layer
#
#                                          PiPVeCC13
# Connection probality of CC excitatory layer 5B cells to PV inhibitory cells of layer 23
########################################################################################################################
fee = 0.125     # multiplication constant for Probablity
fei = 1.0

# some extra factors
FiPVe1x = 1.0
FiPVe2x = 1.0
FiPVe3x = 1.0
FiPVe4x = 1.0

FiSSTe1x = 0.60784313725
FiSSTe2x = 0.60784313725
FiSSTe3x = 0.60784313725
FiSSTe4x = 0.60784313725

FiPViPVxx = 1.0
FiPViSSTxx = 1.0
FiSSTiPVxx = 1.0
FiSSTiSSTxx = 1.0
FiVIPiVIPxx = 1.0
FiVIPiSSTxx = 1.0
FiVIPiPVxx = 1.0
FiSSTiVIPxx = 1.0
FiPViVIPxx = 1.0


PiPVe11 = (((0.745024797305*fee)/(0.745024797305*fee)))*FiPVe1x                 # (Pee11/Pe11)*FiPVe1x
PiPVe12 = (((0.596575792815*fee)/(0.745024797305*fee)))*FiPVe1x                 # (Pee12/Pe11)*FiPVe1x
PiPVe13 = (((0.165500673261*fee)/(0.745024797305*fee)))*FiPVe1x                 # (Pee13/Pe11)*FiPVe1x
PiPVe14 = (((0.0262828621036*fee)/(0.745024797305*fee)))*FiPVe1x                # (Pee14/Pe11)*FiPVe1x

PiPVe41 = (((0.0229751918547*fee)/(0.440617163968*fee)))*FiPVe4x                # (Pee41/Pe44)*FiPVe4x
PiPVe42 = (((0.0441277542125*fee)/(0.440617163968*fee)))*FiPVe4x                # (Pee42/Pe44)*FiPVe4x
PiPVe43 = (((0.164220866279*fee)/(0.440617163968*fee)))*FiPVe4x                 # (Pee43/Pe44)*FiPVe4x
PiPVe44 = (((0.440617163968*fee)/(0.440617163968*fee)))*FiPVe4x                 # (Pee44/Pe44)*FiPVe4x

PiSSTe41 = (((0.0229751918547*fee)/(0.440617163968*fee)))*FiSSTe4x                # (Pee41/Pe44)*FiSSTe4x
PiSSTe42 = (((0.0441277542125*fee)/(0.440617163968*fee)))*FiSSTe4x                # (Pee42/Pe44)*FiSSTe4x
PiSSTe43 = (((0.164220866279*fee)/(0.440617163968*fee)))*FiSSTe4x                 # (Pee43/Pe44)*FiSSTe4x
PiSSTe44 = (((0.440617163968*fee)/(0.440617163968*fee)))*FiSSTe4x                 # (Pee44/Pe44)*FiSSTe4x

etoi_scale =0.25
WiSBCixx = 0.00                  # except for layer 1
WiSBCexx = 0.00
WiENGCixx = 0.00                 # except for layer 1
WiENGCexx = 0.00
WiPVexx = 0.82 * etoi_scale
WiSSTexx = 0.39 * etoi_scale
WiSSTixx = 0.00                   #except for layer 1
WiVIPixx = 0.00                   #except for layer 1
WiVIPexx = 0.39 * etoi_scale
WiPViPVxx = -1.12
WiPViSSTxx = -0.74
WiPViVIPxx = 0.0

Weexx = -0.72
WeiSBC = 0.00
Weixx = -1.00                    # except for the SBC inhibitory neurons

De =  1.5

connect_matrix = {
    'Probability':{
        "PeCCeCC11": 0.745024797305*fee,   "PeCCeCS12": 0.596575792815*fee,  "PeCCeCC12": 0.596575792815*fee,  "PeCCeCT12": 0.596575792815*fee,
        "PeCCeCS13": 0.165500673261*fee,  "PeCCeCC13": 0.165500673261*fee,  "PeCCePT13": 0.165500673261*fee,  "PeCCeCT14": 0.0262828621036*fee,
        "PeCCiSBC10": 0.0, "PeCCiENGC10": 0.64947291, "PeCCiPV11": 0.602984130998*fei, "PeCCiSST11": 0.602984130998*fei,"PeCCiVIP11": 0.41*fei,
        "PeCCiPV12": 0.124183006536*fei,"PeCCiSST12": 0.124183006536*fei,"PeCCiPV13": 0.0915032679739*fei,"PeCCiSST13": 0.0915032679739*fei,"PeCCiPV14": 0.0*fei,"PeCCiSST14": 0.0*fei,

        "PeCSeCC21": 0.918919524474*fee,   "PeCSeCS22": 0.846730004856*fee,  "PeCSeCC22": 0.846730004856*fee,  "PeCSeCT22": 0.846730004856*fee,
        "PeCSeCS23": 0.196623415999*fee,  "PeCSeCC23": 0.196623415999*fee,  "PeCSePT23": 0.196623415999*fee,  "PeCSeCT24": 0.0394450427833*fee,
        "PeCSiSBC20": 0.0, "PeCSiENGC20": 0.26, "PeCSiPV21": 0.0244346510601*fei, "PeCSiSST21": 0.0244346510601*fei,"PeCSiVIP21": 0.0*fei,
        "PeCSiPV22": 0.636601307193*fei,"PeCSiSST22": 0.636601307193*fei,"PeCSiPV23": 0.596078431373*fei,"PeCSiSST23": 0.596078431373*fei,"PeCSiPV24": 0.008898271007*fei,"PeCSiSST24": 0.008898271007*fei,

        "PeCCeCC21": 0.918919524474*fee,   "PeCCeCS22": 0.846730004856*fee,  "PeCCeCC22": 0.846730004856*fee,  "PeCCeCT22": 0.846730004856*fee,
        "PeCCeCS23": 0.196623415999*fee,  "PeCCeCC23": 0.196623415999*fee,  "PeCCePT23": 0.196623415999*fee,  "PeCCeCT24": 0.0394450427833*fee,
        "PeCCiSBC20": 0.0, "PeCCiENGC20": 0.26, "PeCCiPV21": 0.0244346510601*fei, "PeCCiSST21": 0.0244346510601*fei,"PeCCiVIP21": 0.0*fei,
        "PeCCiPV22": 0.636601307193*fei,"PeCCiSST22": 0.636601307193*fei,"PeCCiPV23": 0.596078431373*fei,"PeCCiSST23": 0.596078431373*fei,"PeCCiPV24": 0.008898271007*fei,"PeCCiSST24": 0.008898271007*fei,

        "PeCTeCC21": 0.918919524474*fee,   "PeCTeCS22": 0.846730004856*fee,  "PeCTeCC22": 0.846730004856*fee,  "PeCTeCT22": 0.846730004856*fee,
        "PeCTeCS23": 0.196623415999*fee,  "PeCTeCC23": 0.196623415999*fee,  "PeCTePT23": 0.196623415999*fee,  "PeCTeCT24": 0.0394450427833*fee,
        "PeCTiSBC20": 0.0, "PeCTiENGC20": 0.26, "PeCTiPV21": 0.0244346510601*fei, "PeCTiSST21": 0.0244346510601*fei,"PeCTiVIP21": 0.0*fei,
        "PeCTiPV22": 0.636601307193*fei,"PeCTiSST22": 0.636601307193*fei,"PeCTiPV23": 0.596078431373*fei,"PeCTiSST23": 0.596078431373*fei,"PeCTiPV24": 0.008898271007*fei,"PeCTiSST24": 0.008898271007*fei,

        "PeCSeCC31": 0.522313482169*fee,   "PeCSeCS32": 0.425058867961*fee,  "PeCSeCC32":0.425058867961*fee,  "PeCSeCT32":0.425058867961*fee,
        "PeCSeCS33": 1.0*fee,  "PeCSeCC33": 1.0*fee,  "PeCSePT33": 1.0*fee,  "PeCSeCT34": 0.299536023905*fee,
        "PeCSiSBC30": 0.0, "PeCSiENGC30": 0.2574876, "PeCSiPV31": 0.0031528582013*fei, "PeCSiSST31": 0.0031528582013*fei,"PeCSiVIP31": 0.0*fei,
        "PeCSiPV32": 0.0522875816993*fei,"PeCSiSST32": 0.0522875816993*fei,"PeCSiPV33": 1.0*fei,"PeCSiSST33": 1.0*fei,"PeCSiPV34": 0.169956976234*fei,"PeCSiSST34": 0.169956976234*fei,

        "PeCCeCC31": 0.522313482169*fee,   "PeCCeCS32": 0.425058867961*fee,  "PeCCeCC32": 0.425058867961*fee,  "PeCCeCT32": 0.425058867961*fee,
        "PeCCeCS33": 1.0*fee,  "PeCCeCC33": 1.0*fee,  "PeCCePT33": 1.0*fee,  "PeCCeCT34": 0.299536023905*fee,
        "PeCCiSBC30": 0.0, "PeCCiENGC30": 0.2574876, "PeCCiPV31": 0.0031528582013*fei, "PeCCiSST31": 0.0031528582013*fei,"PeCCiVIP31": 0.0*fei,
        "PeCCiPV32": 0.0522875816993*fei,"PeCCiSST32": 0.0522875816993*fei,"PeCCiPV33": 1.0*fei,"PeCCiSST33": 1.0*fei,"PeCCiPV34": 0.169956976234*fei,"PeCCiSST34": 0.169956976234*fei,

        "PePTeCC31": 0.522313482169*fee,   "PePTeCS32": 0.425058867961*fee,  "PePTeCC32": 0.425058867961*fee,  "PePTeCT32": 0.425058867961*fee,
        "PePTeCS33": 1.0*fee,  "PePTeCC33": 1.0*fee,  "PePTePT33": 1.0*fee,  "PePTeCT34": 0.299536023905*fee,
        "PePTiSBC30": 0.0, "PePTiENGC30": 0.2574876, "PePTiPV31": 0.0031528582013*fei, "PePTiSST31": 0.0031528582013*fei,"PePTiVIP31": 0.0*fei,
        "PePTiPV32": 0.0522875816993*fei,"PePTiSST32": 0.0522875816993*fei,"PePTiPV33": 1.0*fei,"PePTiSST33": 1.0*fei,"PePTiPV34": 0.169956976234*fei,"PePTiSST34": 0.169956976234*fei,

        "PeCTeCC41": 0.0229751918547*fee,   "PeCTeCS42": 0.0441277542125*fee,  "PeCTeCC42": 0.0441277542125*fee,  "PeCTeCT42": 0.0441277542125*fee,
        "PeCTeCS43": 0.164220866279*fee,  "PeCTeCC43": 0.164220866279*fee,  "PeCTePT43": 0.164220866279*fee,  "PeCTeCT44": 0.440617163968*fee,
        "PeCTiSBC40": 0.0, "PeCTiENGC40": 0.0, "PeCTiPV41": 0.00236464365097*fei, "PeCTiSST41": 0.00236464365097*fei,"PeCTiVIP41": 0.0*fei,
        "PeCTiPV42": 0.0*fei,"PeCTiSST42": 0.0*fei,"PeCTiPV43": 0.0287581699346*fei,"PeCTiSST43": 0.0287581699346*fei,"PeCTiPV44": 0.867581423182*fei,"PeCTiSST44": 0.867581423182*fei,





        "PiSBCeCC01": 0.02,   "PiSBCeCS02": 0.0,  "PiSBCeCC02": 0.0,  "PiSBCeCT02": 0.0,  "PiSBCeCS03": 0.0,  "PiSBCeCC03": 0.0,  "PiSBCePT03": 0.0,  "PiSBCeCT04": 0.0,
        "PiSBCiSBC00": 0.13, "PiSBCiENGC00": 1.00, "PiSBCiPV01": 0.0, "PiSBCiSST01": 0.73889088,"PiSBCiVIP01": 0.0,"PiSBCiPV02": 0.0, "PiSBCiSST02": 0.0,
        "PiSBCiPV03": 0.0, "PiSBCiSST03": 0.0,"PiSBCiPV04": 0.0,"PiSBCiSST04": 0.0,

        "PiENGCeCC01": 0.0,   "PiENGCeCS02": 0.0,  "PiENGCeCC02": 0.0,  "PiENGCeCT02": 0.0,  "PiENGCeCS03": 0.0,  "PiENGCeCC03": 0.0,  "PiENGCePT03": 0.0,  "PiENGCeCT04": 0.0,
        "PiENGCiSBC00": 0.0, "PiENGCiENGC00": 1.0, "PiENGCiPV01": 0.0, "PiENGCiSST01": 1.0,"PiENGCiVIP01": 0.0,"PiENGCiPV02": 0.0,"PiENGCiSST02": 0.0,
        "PiENGCiPV03": 0.0,"PiENGCiSST03": 0.0,"PiENGCiPV04": 0.0,"PiENGCiSST04": 0.0,

        "PiPVeCC11": 0.41,   "PiPVeCS12": PiPVe12,  "PiPVeCC12": PiPVe12,  "PiPVeCT12": PiPVe12,  "PiPVeCS13": PiPVe13,  "PiPVeCC13": PiPVe13,  "PiPVePT13": PiPVe13,  "PiPVeCT14": PiPVe14,
        "PiPViSBC10": 0.5, "PiPViENGC10": 0.2568036, "PiPViPV11": 0.602984130998*fei*FiPViPVxx, "PiPViSST11": 0.857*FiPViSSTxx,"PiPViVIP11": 0.0*FiPViVIPxx,
        "PiPViPV12": 0.124183006536*fei*FiPViPVxx, "PiPViSST12": 0.0*FiPViSSTxx,
        "PiPViPV13": 0.0915032679739*fei*FiPViPVxx, "PiPViSST13": 0.0*FiPViSSTxx,"PiPViPV14": 0.0*fei*FiPViPVxx,"PiPViSST14": 0.0*FiPViSSTxx,

        "PiSSTeCC11": 0.41,   "PiSSTeCS12": 0.0*FiSSTe1x,  "PiSSTeCC12": 0.0*FiSSTe1x,  "PiSSTeCT12": 0.0*FiSSTe1x,
        "PiSSTeCS13": 0.0*FiSSTe1x,  "PiSSTeCC13": 0.0*FiSSTe1x,  "PiSSTePT13": 0.0*FiSSTe1x,  "PiSSTeCT14": 0.0*FiSSTe1x,
        "PiSSTiSBC10": 0.0, "PiSSTiENGC10": 0.0, "PiSSTiPV11": 0.41, "PiSSTiSST11": 0.0*FiSSTiSSTxx,"PiSSTiVIP11": 0.616*FiSSTiVIPxx,
        "PiSSTiPV12": 0.0*FiSSTiPVxx,"PiSSTiSST12": 0.0*FiSSTiSSTxx,
        "PiSSTiPV13": 0.0*FiSSTiPVxx,"PiSSTiSST13": 0.0*FiSSTiSSTxx,"PiSSTiPV14": 0.0*FiSSTiPVxx,"PiSSTiSST14": 0.0*FiSSTiSSTxx,

        "PiVIPeCC11": 0.55, "PiVIPeCS12": 0.0,  "PiVIPeCC12": 0.0,  "PiVIPeCT12": 0.0,  "PiVIPeCS13": 0.0,  "PiVIPeCC13": 0.0,  "PiVIPePT13": 0.0,  "PiVIPeCT14": 0.0,
        "PiVIPiSBC10": 0.43, "PiVIPiENGC10": 0.8, "PiVIPiPV11": 0.28, "PiVIPiSST11": 0.616*FiVIPiSSTxx,"PiVIPiVIP11": 0.3*FiVIPiVIPxx,
        "PiVIPiPV12": 0.0*FiVIPiPVxx, "PiVIPiSST12": 0.0*FiVIPiSSTxx, "PiVIPiPV13": 0.0*FiVIPiPVxx,"PiVIPiSST13": 0.0*FiVIPiSSTxx, "PiVIPiPV14": 0.0*FiVIPiPVxx,"PiVIPiSST14": 0.0*FiVIPiSSTxx,

        "PiPVeCC21":0.632014795996*FiPVe2x,   "PiPVeCS22": 1.0*FiPVe2x,  "PiPVeCC22": 1.0*FiPVe2x,  "PiPVeCT22": 1.0*FiPVe2x,
        "PiPVeCS23": 0.470111504179*FiPVe2x,  "PiPVeCC23": 0.470111504179*FiPVe2x,  "PiPVePT23": 0.470111504179*FiPVe2x,  "PiPVeCT24": 0.0710308221417*FiPVe2x,
        "PiPViSBC20": 0.0, "PiPViENGC20": 0.0, "PiPViPV21": 0.0244346510601*fei*FiPViPVxx, "PiPViSST21": 0.0*FiPViSSTxx,"PiPViVIP21": 0.0*FiPViVIPxx,"PiPViPV22": 0.636601307193*fei*FiPViPVxx,"PiPViSST22": 0.857*FiPViSSTxx,
        "PiPViPV23": 0.596078431373*fei*FiPViPVxx,"PiPViSST23": 0.0*FiPViSSTxx,"PiPViPV24": 0.008898271007*fei*FiPViPVxx,"PiPViSST24": 0.0*FiPViSSTxx,

        "PiSSTeCC21": 1.0*FiSSTe2x,   "PiSSTeCS22": 0.587835016033*FiSSTe2x,  "PiSSTeCC22": 0.587835016033*FiSSTe2x,  "PiSSTeCT22": 0.587835016033*FiSSTe2x,
        "PiSSTeCS23": 0.178087060371*FiSSTe2x,  "PiSSTeCC23": 0.178087060371*FiSSTe2x,  "PiSSTePT23": 0.178087060371*FiSSTe2x,  "PiSSTeCT24": 0.0627023068196*FiSSTe2x,
        "PiSSTiSBC20": 0.0, "PiSSTiENGC20": 0.0, "PiSSTiPV21": 0.0*FiSSTiPVxx, "PiSSTiSST21": 0.0*FiSSTiSSTxx,"PiSSTiVIP21": 0.616*FiSSTiVIPxx,"PiSSTiPV22": 0.0*FiSSTiPVxx,"PiSSTiSST22": 0.0*FiSSTiSSTxx,
        "PiSSTiPV23": 0.0*FiSSTiPVxx,"PiSSTiSST23": 0.0*FiSSTiSSTxx,"PiSSTiPV24": 0.0*FiSSTiPVxx,"PiSSTiSST24": 0.0*FiSSTiSSTxx,

        "PiPVeCC31": 0.2764505774*FiPVe3x,   "PiPVeCS32": 1.0*FiPVe3x,  "PiPVeCC32": 1.0*FiPVe3x,  "PiPVeCT32": 1.0*FiPVe3x,
        "PiPVeCS33": 0.790285463718*FiPVe3x,  "PiPVeCC33": 0.790285463718*FiPVe3x,  "PiPVePT33": 0.790285463718*FiPVe3x,  "PiPVeCT34": 0.169267347837*FiPVe3x,
        "PiPViSBC30": 0.0, "PiPViENGC30": 0.0, "PiPViPV31": 0.0031528582013*fei*FiPViPVxx, "PiPViSST31": 0.0*FiPViSSTxx, "PiPViVIP31": 0.0*FiPViVIPxx,
        "PiPViPV32": 0.0522875816993*fei*FiPViPVxx,"PiPViSST32": 0.0*FiPViSSTxx,
        "PiPViPV33": 1.0*fei*FiPViPVxx,"PiPViSST33": 0.857*FiPViSSTxx,"PiPViPV34": 0.169956976234*fei*FiPViPVxx,"PiPViSST34": 0.0*FiPViSSTxx,

        "PiSSTeCC31": 1.0*FiSSTe3x,   "PiSSTeCS32": 0.443956673808*FiSSTe3x,  "PiSSTeCC32": 0.443956673808*FiSSTe3x,  "PiSSTeCT32": 0.443956673808*FiSSTe3x,
        "PiSSTeCS33": 0.476504960848*FiSSTe3x,  "PiSSTeCC33": 0.476504960848*FiSSTe3x,  "PiSSTePT33": 0.476504960848*FiSSTe3x,  "PiSSTeCT34": 0.183158759158*FiSSTe3x,
        "PiSSTiSBC30": 0.0, "PiSSTiENGC30": 0.0, "PiSSTiPV31": 0.0*FiSSTiPVxx, "PiSSTiSST31": 0.0*FiSSTiSSTxx,"PiSSTiVIP31": 0.616*FiSSTiVIPxx,"PiSSTiPV32": 0.0*FiSSTiPVxx,"PiSSTiSST32": 0.0*FiSSTiSSTxx,
        "PiSSTiPV33": 0.0*FiSSTiPVxx,"PiSSTiSST33": 0.0*FiSSTiSSTxx,"PiSSTiPV34": 0.0*FiSSTiPVxx,"PiSSTiSST34": 0.0*FiSSTiSSTxx,

        "PiPVeCC41": PiPVe41,   "PiPVeCS42": PiPVe42,  "PiPVeCC42": PiPVe42,  "PiPVeCT42": PiPVe42,  "PiPVeCS43": PiPVe43,  "PiPVeCC43": PiPVe43,  "PiPVePT43": PiPVe43,  "PiPVeCT44": PiPVe44,
        "PiPViSBC40": 0.0, "PiPViENGC40": 0.0, "PiPViPV41": 0.00236464365097*fei*FiPViPVxx, "PiPViSST41": 0.0*FiPViSSTxx ,"PiPViVIP41": 0.0*FiPViVIPxx,"PiPViPV42": 0.0*fei*FiPViPVxx,"PiPViSST42":0.0*FiPViSSTxx ,
        "PiPViPV43": 0.0287581699346*fei*FiPViPVxx,"PiPViSST43": 0.0*FiPViSSTxx,"PiPViPV44": 0.867581423182*fei*FiPViPVxx,"PiPViSST44": 0.857*FiPViSSTxx,

        "PiSSTeCC41": PiSSTe41,   "PiSSTeCS42": PiSSTe42,  "PiSSTeCC42": PiSSTe42,  "PiSSTeCT42": PiSSTe42,  "PiSSTeCS43": PiSSTe43,  "PiSSTeCC43": PiSSTe43,  "PiSSTePT43": PiSSTe43,  "PiSSTeCT44": PiSSTe44,
        "PiSSTiSBC40": 0.0, "PiSSTiENGC40": 0.0, "PiSSTiPV41": 0.0*FiSSTiPVxx, "PiSSTiSST41": 0.0*FiSSTiSSTxx,"PiSSTiVIP41": 0.616*FiSSTiVIPxx,"PiSSTiPV42": 0.0*FiSSTiPVxx,"PiSSTiSST42": 0.0*FiSSTiSSTxx,
        "PiSSTiPV43": 0.0*FiSSTiPVxx,"PiSSTiSST43": 0.0*FiSSTiSSTxx,"PiSSTiPV44": 0.0*FiSSTiPVxx,"PiSSTiSST44": 0.0*FiSSTiSSTxx,
        },
    'Sigma':{
        "SeCCeCC11":  467.702817055, "SeCCeCS12": 548.013844352,  "SeCCeCC12": 548.013844352,  "SeCCeCT12": 548.013844352,
        "SeCCeCS13": 606.368829317,  "SeCCeCC13": 606.368829317,  "SeCCePT13": 606.368829317,  "SeCCeCT14": 585.132412528,
        "SeCCiSBC10": 34.349899622, "SeCCiENGC10": 213.490490572,"SeCCiPV11":100.0 ,"SeCCiSST11": 100.0,"SeCCiVIP11": 100.0,
        "SeCCiPV12": 131.0,"SeCCiSST12": 131.0,"SeCCiPV13": 202.0,"SeCCiSST13": 202.0,"SeCCiPV14": 162.0,"SeCCiSST14": 162.0,

        "SeCSeCC21": 467.702817055,  "SeCSeCS22": 548.013844352,  "SeCSeCC22": 548.013844352,  "SeCSeCT22": 548.013844352,
        "SeCSeCS23": 606.368829317,  "SeCSeCC23": 606.368829317,  "SeCSePT23": 606.368829317,  "SeCSeCT24": 585.132412528,
        "SeCSiSBC20": 34.349899622, "SeCSiENGC20": 184.821601654,"SeCSiPV21": 100.0,  "SeCSiSST21": 100.0,"SeCSiVIP21": 100.0,
        "SeCSiPV22": 131.0,"SeCSiSST22": 131.0,"SeCSiPV23": 202.0,"SeCSiSST23": 202.0,"SeCSiPV24": 162.0,"SeCSiSST24": 162.0,

        "SeCCeCC21":467.702817055,   "SeCCeCS22": 548.013844352,  "SeCCeCC22": 548.013844352,  "SeCCeCT22": 548.013844352,
        "SeCCeCS23": 606.368829317, "SeCCeCC23": 606.368829317,"SeCCePT23": 606.368829317, "SeCCeCT24": 585.132412528,
        "SeCCiSBC20": 34.349899622, "SeCCiENGC20": 184.821601654,"SeCCiPV21": 100.0,  "SeCCiSST21": 100.0,"SeCCiVIP21": 100.0,
        "SeCCiPV22": 131.0,"SeCCiSST22": 131.0,"SeCCiPV23": 202.0,"SeCCiSST23": 202.0,"SeCCiPV24": 162.0,"SeCCiSST24": 162.0,

        "SeCTeCC21":467.702817055,   "SeCTeCS22": 548.013844352,  "SeCTeCC22": 548.013844352,  "SeCTeCT22": 548.013844352,
        "SeCTeCS23": 606.368829317, "SeCTeCC23": 606.368829317,"SeCTePT23": 606.368829317, "SeCTeCT24": 585.132412528,
        "SeCTiSBC20": 34.349899622, "SeCTiENGC20": 184.821601654,"SeCTiPV21": 100.0,  "SeCTiSST21": 100.0,"SeCTiVIP21": 100.0,
        "SeCTiPV22": 131.0,"SeCTiSST22": 131.0,"SeCTiPV23": 202.0,"SeCTiSST23": 202.0,"SeCTiPV24": 162.0,"SeCTiSST24": 162.0,

        "SeCSeCC31": 467.702817055,  "SeCSeCS32": 548.013844352,  "SeCSeCC32": 548.013844352,  "SeCSeCT32": 548.013844352,
        "SeCSeCS33": 606.368829317,  "SeCSeCC33": 606.368829317,  "SeCSePT33": 606.368829317, "SeCSeCT34": 585.132412528,
        "SeCSiSBC30": 34.349899622, "SeCSiENGC30": 184.821601654,"SeCSiPV31": 100.0, "SeCSiSST31": 100.0, "SeCSiVIP31": 100.0,
        "SeCSiPV32": 131.0,"SeCSiSST32": 131.0,"SeCSiPV33": 202.0,"SeCSiSST33": 202.0,"SeCSiPV34": 162.0,"SeCSiSST34": 162.0,

        "SeCCeCC31":467.702817055,   "SeCCeCS32": 548.013844352,  "SeCCeCC32": 548.013844352, "SeCCeCT32": 548.013844352,
        "SeCCeCS33": 606.368829317, "SeCCeCC33": 606.368829317,"SeCCePT33": 606.368829317, "SeCCeCT34": 585.132412528,
        "SeCCiSBC30": 34.349899622, "SeCCiENGC30": 184.821601654,"SeCCiPV31": 100.0, "SeCCiSST31": 100.0, "SeCCiVIP31": 100.0,
        "SeCCiPV32": 131.0,"SeCCiSST32": 131.0,"SeCCiPV33": 202.0,"SeCCiSST33": 202.0,"SeCCiPV34": 162.0,"SeCCiSST34": 162.0,

        "SePTeCC31": 467.702817055,  "SePTeCS32": 548.013844352, "SePTeCC32": 548.013844352, "SePTeCT32": 548.013844352,
        "SePTeCS33": 606.368829317, "SePTeCC33": 606.368829317,"SePTePT33": 606.368829317, "SePTeCT34": 585.132412528,
        "SePTiSBC30": 34.349899622, "SePTiENGC30": 184.821601654,"SePTiPV31": 100.0, "SePTiSST31": 100.0, "SePTiVIP31": 100.0,
        "SePTiPV32": 131.0,"SePTiSST32": 131.0,"SePTiPV33": 202.0,"SePTiSST33": 202.0,"SePTiPV34": 162.0,"SePTiSST34": 162.0,

        "SeCTeCC41": 467.702817055,  "SeCTeCS42": 548.013844352,  "SeCTeCC42": 548.013844352, "SeCTeCT42": 548.013844352,
        "SeCTeCS43": 606.368829317,  "SeCTeCC43": 606.368829317,  "SeCTePT43": 606.368829317, "SeCTeCT44": 585.132412528,
        "SeCTiSBC40": 34.349899622, "SeCTiENGC40": 184.821601654,"SeCTiPV41": 100.0, "SeCTiSST41": 100.0, "SeCTiVIP41": 100.0,
        "SeCTiPV42": 131.0,"SeCTiSST42": 131.0,"SeCTiPV43": 202.0,"SeCTiSST43": 202.0,"SeCTiPV44": 162.0,"SeCTiSST44": 162.0,


        "SiSBCeCC01": 467.702817055,  "SiSBCeCS02":  548.013844352,  "SiSBCeCC02":  548.013844352, "SiSBCeCT02": 548.013844352,
        "SiSBCeCS03": 606.368829317,  "SiSBCeCC03": 606.368829317,  "SiSBCePT03": 606.368829317,  "SiSBCeCT04": 585.132412528,
        "SiSBCiSBC00": 34.349899622, "SiSBCiENGC00": 238.794855759, "SiSBCiPV01": 100.0,  "SiSBCiSST01":  100.0,"SiSBCiVIP01": 100.0,
        "SiSBCiPV02": 131.0, "SiSBCiSST02": 131.0, "SiSBCiPV03": 202.0,  "SiSBCiSST03": 202.0, "SiSBCiPV04": 162.0,  "SiSBCiSST04": 162.0,

        "SiENGCeCC01": 467.702817055,  "SiENGCeCS02": 548.013844352,  "SiENGCeCC02": 548.013844352, "SiENGCeCT02": 548.013844352,
        "SiENGCeCS03": 606.368829317, "SiENGCeCC03": 606.368829317, "SiENGCePT03": 606.368829317, "SiENGCeCT04": 585.132412528,
        "SiENGCiSBC00": 34.349899622,"SiENGCiENGC00": 238.794855759,"SiENGCiPV01": 100.0, "SiENGCiSST01": 100.0,"SiENGCiVIP01": 100.0,
        "SiENGCiPV02": 131.0, "SiENGCiSST02": 131.0,"SiENGCiPV03": 202.0, "SiENGCiSST03": 202.0,"SiENGCiPV04": 162.0,"SiENGCiSST04": 162.0,

        "SiPVeCC11":467.702817055,    "SiPVeCS12": 548.013844352,    "SiPVeCC12": 548.013844352,   "SiPVeCT12": 548.013844352,
        "SiPVeCS13": 606.368829317,   "SiPVeCC13": 606.368829317,   "SiPVePT13": 606.368829317,   "SiPVeCT14": 585.132412528,
        "SiPViSBC10": 34.349899622,  "SiPViENGC10": 238.794855759,  "SiPViPV11": 100.0,   "SiPViSST11": 100.0,  "SiPViVIP11": 100.0,
        "SiPViPV12": 131.0,   "SiPViSST12": 131.0,  "SiPViPV13": 202.0,   "SiPViSST13": 202.0,  "SiPViPV14": 162.0,  "SiPViSST14": 162.0,

        "SiSSTeCC11":467.702817055,   "SiSSTeCS12": 548.013844352,   "SiSSTeCC12": 548.013844352,  "SiSSTeCT12": 548.013844352,
        "SiSSTeCS13": 606.368829317,  "SiSSTeCC13": 606.368829317,  "SiSSTePT13": 606.368829317,  "SiSSTeCT14": 585.132412528,
        "SiSSTiSBC10": 34.349899622, "SiSSTiENGC10": 238.794855759, "SiSSTiPV11": 100.0,  "SiSSTiSST11": 100.0, "SiSSTiVIP11": 100.0,
        "SiSSTiPV12": 131.0,  "SiSSTiSST12": 131.0, "SiSSTiPV13": 202.0,  "SiSSTiSST13": 202.0, "SiSSTiPV14": 162.0, "SiSSTiSST14": 162.0,

        "SiVIPeCC11": 467.702817055,   "SiVIPeCS12": 548.013844352,   "SiVIPeCC12": 548.013844352,  "SiVIPeCT12": 548.013844352,
        "SiVIPeCS13": 606.368829317,  "SiVIPeCC13": 606.368829317,  "SiVIPePT13": 606.368829317,  "SiVIPeCT14": 585.132412528,
        "SiVIPiSBC10": 34.349899622, "SiVIPiENGC10": 238.794855759, "SiVIPiPV11": 100.0,  "SiVIPiSST11": 100.0, "SiVIPiVIP11": 100.0,
        "SiVIPiPV12": 131.0,  "SiVIPiSST12": 131.0, "SiVIPiPV13": 202.0,  "SiVIPiSST13": 202.0, "SiVIPiPV14": 162.0, "SiVIPiSST14": 162.0,

        "SiPVeCC21": 361.0,    "SiPVeCS22": 344.0,    "SiPVeCC22": 344.0,   "SiPVeCT22": 344.0, "SiPVeCS23": 492.0, "SiPVeCC23": 492.0, "SiPVePT23": 492.0,   "SiPVeCT24": 839.,
        "SiPViSBC20": 34.349899622,  "SiPViENGC20": 238.794855759,  "SiPViPV21": 100.0,   "SiPViSST21": 100.0,  "SiPViVIP21": 100.0,
        "SiPViPV22": 131.0,   "SiPViSST22": 131.0,  "SiPViPV23": 202.0,   "SiPViSST23": 202.0,  "SiPViPV24": 162.0,  "SiPViSST24": 162.0,

        "SiSSTeCC21":471.0,   "SiSSTeCS22": 394.0,   "SiSSTeCC22": 394.0,  "SiSSTeCT22": 394.0,  "SiSSTeCS23": 734.0,  "SiSSTeCC23": 734.0,  "SiSSTePT23": 734.0,  "SiSSTeCT24": 1110.0,
        "SiSSTiSBC20": 34.349899622, "SiSSTiENGC20": 238.794855759, "SiSSTiPV21": 100.0,  "SiSSTiSST21": 100.0, "SiSSTiVIP21": 100.0,
        "SiSSTiPV22": 131.0,  "SiSSTiSST22": 131.0, "SiSSTiPV23": 202.0,  "SiSSTiSST23": 202.0, "SiSSTiPV24": 162.0, "SiSSTiSST24": 162.0,

        "SiPVeCC31": 611.0,    "SiPVeCS32": 240.0,    "SiPVeCC32": 240.0,   "SiPVeCT32": 240.0,   "SiPVeCS33": 421.0,   "SiPVeCC33": 421.0,   "SiPVePT33": 421.0,   "SiPVeCT34": 647.0,
        "SiPViSBC30": 34.349899622,  "SiPViENGC30": 238.794855759,  "SiPViPV31": 100.0,   "SiPViSST31": 100.0,  "SiPViVIP31": 100.0,
        "SiPViPV32": 131.0,   "SiPViSST32": 131.0,  "SiPViPV33": 202.0,   "SiPViSST33": 202.0,  "SiPViPV34": 162.0,  "SiPViSST34": 162.0,

        "SiSSTeCC31":469.0,   "SiSSTeCS32": 601.0,   "SiSSTeCC32": 601.0,  "SiSSTeCT32": 601.0,  "SiSSTeCS33": 597.0,  "SiSSTeCC33": 597.0,  "SiSSTePT33": 597.0,  "SiSSTeCT34": 1091.0,
        "SiSSTiSBC30": 34.349899622, "SiSSTiENGC30": 238.794855759, "SiSSTiPV31": 100.0,  "SiSSTiSST31": 100.0, "SiSSTiVIP31": 100.0,
        "SiSSTiPV32": 131.0,  "SiSSTiSST32": 131.0, "SiSSTiPV33": 202.0,  "SiSSTiSST33": 202.0, "SiSSTiPV34": 162.0, "SiSSTiSST34": 162.0,

        "SiPVeCC41":467.702817055,    "SiPVeCS42": 548.013844352,    "SiPVeCC42": 548.013844352,   "SiPVeCT42": 548.013844352,
        "SiPVeCS43": 606.368829317,   "SiPVeCC43": 606.368829317,   "SiPVePT43": 606.368829317,   "SiPVeCT44": 585.132412528,
        "SiPViSBC40": 34.349899622,  "SiPViENGC40": 238.794855759,  "SiPViPV41": 100.0,   "SiPViSST41": 100.0,  "SiPViVIP41": 100.0,
        "SiPViPV42": 131.0,   "SiPViSST42": 131.0,  "SiPViPV43": 202.0,   "SiPViSST43": 202.0,  "SiPViPV44": 162.0,  "SiPViSST44": 162.0,

        "SiSSTeCC41":467.702817055,   "SiSSTeCS42": 548.013844352,   "SiSSTeCC42": 548.013844352,  "SiSSTeCT42": 548.013844352,
        "SiSSTeCS43": 606.368829317,  "SiSSTeCC43": 606.368829317,  "SiSSTePT43": 606.368829317,  "SiSSTeCT44": 585.132412528,
        "SiSSTiSBC40": 34.349899622, "SiSSTiENGC40": 238.794855759, "SiSSTiPV41": 100.0,  "SiSSTiSST41": 100.0, "SiSSTiVIP41": 100.0,
        "SiSSTiPV42": 131.0,  "SiSSTiSST42": 131.0, "SiSSTiPV43": 202.0,  "SiSSTiSST43": 202.0, "SiSSTiPV44": 162.0, "SiSSTiSST44": 162.0,
        },
    'Weight':{
        "WeCCeCC11": Weexx, "WeCCeCS12": Weexx,  "WeCCeCC12": Weexx,  "WeCCeCT12": Weexx, "WeCCeCS13": Weexx,  "WeCCeCC13": Weexx,  "WeCCePT13": Weexx,  "WeCCeCT14": Weexx,
        "WeCCiSBC10": WeiSBC, "WeCCiENGC10": Weixx,"WeCCiPV11": Weixx,  "WeCCiSST11": Weixx,"WeCCiVIP11": Weixx,"WeCCiPV12": Weixx,"WeCCiSST12": Weixx,
        "WeCCiPV13": Weixx,"WeCCiSST13": Weixx,"WeCCiPV14": Weixx,"WeCCiSST14":Weixx,

        "WeCSeCC21": Weexx,  "WeCSeCS22": Weexx,  "WeCSeCC22": Weexx,  "WeCSeCT22": Weexx, "WeCSeCS23": Weexx,  "WeCSeCC23": Weexx,  "WeCSePT23": Weexx,  "WeCSeCT24": Weexx,
        "WeCSiSBC20": WeiSBC, "WeCSiENGC20": Weixx,"WeCSiPV21": Weixx,  "WeCSiSST21": Weixx,"WeCSiVIP21": Weixx,"WeCSiPV22": Weixx,"WeCSiSST22": Weixx,
        "WeCSiPV23": Weixx,"WeCSiSST23": Weixx,"WeCSiPV24": Weixx,"WeCSiSST24": Weixx,

        "WeCCeCC21":Weexx,   "WeCCeCS22": Weexx,  "WeCCeCC22": Weexx,  "WeCCeCT22": Weexx, "WeCCeCS23": Weexx, "WeCCeCC23": Weexx,"WeCCePT23": Weexx, "WeCCeCT24": Weexx,
        "WeCCiSBC20": WeiSBC, "WeCCiENGC20": Weixx,"WeCCiPV21": Weixx,  "WeCCiSST21": Weixx,"WeCCiVIP21": Weixx,"WeCCiPV22": Weixx,"WeCCiSST22": Weixx,
        "WeCCiPV23": Weixx,"WeCCiSST23": Weixx,"WeCCiPV24":Weixx,"WeCCiSST24": Weixx,

        "WeCTeCC21": Weexx,   "WeCTeCS22": Weexx,  "WeCTeCC22": Weexx,  "WeCTeCT22": Weexx, "WeCTeCS23": Weexx, "WeCTeCC23": Weexx,"WeCTePT23": Weexx, "WeCTeCT24": Weexx,
        "WeCTiSBC20": WeiSBC, "WeCTiENGC20": Weixx,"WeCTiPV21": Weixx,  "WeCTiSST21": Weixx,"WeCTiVIP21": Weixx,"WeCTiPV22": Weixx,"WeCTiSST22": Weixx,
        "WeCTiPV23": Weixx,"WeCTiSST23": Weixx,"WeCTiPV24": Weixx,"WeCTiSST24": Weixx,

        "WeCSeCC31": Weexx,  "WeCSeCS32": Weexx,  "WeCSeCC32": Weexx,  "WeCSeCT32":Weexx, "WeCSeCS33": Weexx,  "WeCSeCC33": Weexx,  "WeCSePT33": Weexx, "WeCSeCT34": Weexx,
        "WeCSiSBC30": WeiSBC, "WeCSiENGC30": Weixx,"WeCSiPV31": Weixx, "WeCSiSST31": Weixx, "WeCSiVIP31": Weixx,"WeCSiPV32": Weixx,"WeCSiSST32": Weixx,
        "WeCSiPV33": Weixx,"WeCSiSST33": Weixx,"WeCSiPV34": Weixx,"WeCSiSST34": Weixx,

        "WeCCeCC31": Weexx,   "WeCCeCS32": Weexx,  "WeCCeCC32": Weexx, "WeCCeCT32": Weexx,  "WeCCeCS33": Weexx, "WeCCeCC33": Weexx,"WeCCePT33": Weexx, "WeCCeCT34": Weexx,
        "WeCCiSBC30": WeiSBC, "WeCCiENGC30": Weixx,"WeCCiPV31": Weixx, "WeCCiSST31": Weixx, "WeCCiVIP31": Weixx,"WeCCiPV32": Weixx,"WeCCiSST32": Weixx,
        "WeCCiPV33": Weixx,"WeCCiSST33": Weixx,"WeCCiPV34": Weixx,"WeCCiSST34": Weixx,

        "WePTeCC31": Weexx,  "WePTeCS32":  Weexx, "WePTeCC32": Weexx, "WePTeCT32": Weexx,   "WePTeCS33": Weexx, "WePTeCC33": Weexx,"WePTePT33": Weexx, "WePTeCT34": Weexx,
        "WePTiSBC30": WeiSBC, "WePTiENGC30": Weixx,"WePTiPV31": Weixx, "WePTiSST31": Weixx, "WePTiVIP31": Weixx,"WePTiPV32": Weixx,"WePTiSST32": Weixx,
        "WePTiPV33": Weixx,"WePTiSST33": Weixx,"WePTiPV34": Weixx,"WePTiSST34": Weixx,

        "WeCTeCC41": Weexx,  "WeCTeCS42": Weexx,  "WeCTeCC42": Weexx, "WeCTeCT42": Weexx, "WeCTeCS43": Weexx,  "WeCTeCC43": Weexx,  "WeCTePT43": Weexx, "WeCTeCT44": Weexx,
        "WeCTiSBC40": WeiSBC, "WeCTiENGC40": Weixx,"WeCTiPV41": Weixx, "WeCTiSST41": Weixx, "WeCTiVIP41": Weixx,"WeCTiPV42": Weixx,"WeCTiSST42": Weixx,
        "WeCTiPV43": Weixx,"WeCTiSST43": Weixx,"WeCTiPV44": Weixx,"WeCTiSST44": Weixx,


        "WiSBCeCC01": WiSBCexx,  "WiSBCeCS02":  WiSBCexx,  "WiSBCeCC02": WiSBCexx, "WiSBCeCT02": WiSBCexx,  "WiSBCeCS03": WiSBCexx,  "WiSBCeCC03": WiSBCexx,  "WiSBCePT03": WiSBCexx,  "WiSBCeCT04": WiSBCexx,
        "WiSBCiSBC00": -0.8, "WiSBCiENGC00": -1.0, "WiSBCiPV01": WiSBCixx,  "WiSBCiSST01":WiSBCixx,"WiSBCiVIP01": WiSBCixx, "WiSBCiPV02": WiSBCixx,
        "WiSBCiSST02": WiSBCixx, "WiSBCiPV03": WiSBCixx,    "WiSBCiSST03": WiSBCixx, "WiSBCiPV04": WiSBCixx,  "WiSBCiSST04": WiSBCixx,

        "WiENGCeCC01":WiENGCexx,  "WiENGCeCS02": WiENGCexx,  "WiENGCeCC02": WiENGCexx, "WiENGCeCT02": WiENGCexx, "WiENGCeCS03": WiENGCexx, "WiENGCeCC03": WiENGCexx, "WiENGCePT03": WiENGCexx, "WiENGCeCT04": WiENGCexx,
        "WiENGCiSBC00": -0.8,"WiENGCiENGC00": -1.0,"WiENGCiPV01": WiENGCixx, "WiENGCiSST01": WiENGCixx,"WiENGCiVIP01": WiENGCixx,"WiENGCiPV02": WiENGCixx, "WiENGCiSST02": WiENGCixx,
        "WiENGCiPV03": WiENGCixx, "WiENGCiSST03": WiENGCixx,"WiENGCiPV04": WiENGCixx,"WiENGCiSST04": WiENGCixx,

        "WiPVeCC11": WiPVexx,    "WiPVeCS12": WiPVexx,    "WiPVeCC12": WiPVexx,   "WiPVeCT12": WiPVexx,   "WiPVeCS13": WiPVexx,   "WiPVeCC13": WiPVexx,   "WiPVePT13": WiPVexx,   "WiPVeCT14": WiPVexx,
        "WiPViSBC10": -0.8,  "WiPViENGC10": -1.0,  "WiPViPV11": WiPViPVxx,   "WiPViSST11": WiPViSSTxx,  "WiPViVIP11": WiPViVIPxx,  "WiPViPV12": WiPViPVxx,   "WiPViSST12": WiPViSSTxx,
        "WiPViPV13": WiPViPVxx,   "WiPViSST13": WiPViSSTxx,  "WiPViPV14": WiPViPVxx,  "WiPViSST14": WiPViSSTxx,

        "WiSSTeCC11":WiSSTexx,   "WiSSTeCS12": WiSSTexx,   "WiSSTeCC12": WiSSTexx,  "WiSSTeCT12": WiSSTexx,  "WiSSTeCS13": WiSSTexx,  "WiSSTeCC13": WiSSTexx,  "WiSSTePT13": WiSSTexx,  "WiSSTeCT14": WiSSTexx,
        "WiSSTiSBC10": -0.8, "WiSSTiENGC10": -1.0, "WiSSTiPV11": WiSSTixx,  "WiSSTiSST11": WiSSTixx, "WiSSTiVIP11": -1.0, "WiSSTiPV12": WiSSTixx,  "WiSSTiSST12": WiSSTixx,
        "WiSSTiPV13": WiSSTixx,  "WiSSTiSST13": WiSSTixx, "WiSSTiPV14": WiSSTixx, "WiSSTiSST14": WiSSTixx,

        "WiVIPeCC11":WiVIPexx,   "WiVIPeCS12": WiVIPexx,   "WiVIPeCC12": WiVIPexx,  "WiVIPeCT12": WiVIPexx,  "WiVIPeCS13": WiVIPexx,  "WiVIPeCC13": WiVIPexx,  "WiVIPePT13": WiVIPexx,  "WiVIPeCT14": WiVIPexx,
        "WiVIPiSBC10": -0.8, "WiVIPiENGC10": -1.0, "WiVIPiPV11": WiVIPixx,  "WiVIPiSST11": -1.0, "WiVIPiVIP11": -1.0, "WiVIPiPV12": WiVIPixx,  "WiVIPiSST12": WiVIPixx,
        "WiVIPiPV13": WiVIPixx,  "WiVIPiSST13": WiVIPixx, "WiVIPiPV14": WiVIPixx, "WiVIPiSST14": WiVIPixx,

        "WiPVeCC21":WiPVexx,    "WiPVeCS22": WiPVexx,    "WiPVeCC22": WiPVexx,   "WiPVeCT22": WiPVexx,   "WiPVeCS23": WiPVexx,   "WiPVeCC23": WiPVexx,   "WiPVePT23": WiPVexx,   "WiPVeCT24": WiPVexx,
        "WiPViSBC20": -0.8,  "WiPViENGC20": -1.0,  "WiPViPV21": WiPViPVxx,   "WiPViSST21": WiPViSSTxx,  "WiPViVIP21": WiPViVIPxx,  "WiPViPV22": WiPViPVxx,   "WiPViSST22": WiPViSSTxx,
        "WiPViPV23": WiPViPVxx,   "WiPViSST23": WiPViSSTxx,  "WiPViPV24": WiPViPVxx,  "WiPViSST24": WiPViSSTxx,

        "WiSSTeCC21":WiSSTexx,   "WiSSTeCS22": WiSSTexx,   "WiSSTeCC22": WiSSTexx,  "WiSSTeCT22": WiSSTexx,  "WiSSTeCS23": WiSSTexx,  "WiSSTeCC23": WiSSTexx,  "WiSSTePT23": WiSSTexx,  "WiSSTeCT24": WiSSTexx,
        "WiSSTiSBC20": -0.8, "WiSSTiENGC20": -1.0, "WiSSTiPV21": WiSSTixx,  "WiSSTiSST21": WiSSTixx, "WiSSTiVIP21": -1.0, "WiSSTiPV22": WiSSTixx,  "WiSSTiSST22": WiSSTixx,
        "WiSSTiPV23": WiSSTixx,  "WiSSTiSST23": WiSSTixx, "WiSSTiPV24": WiSSTixx, "WiSSTiSST24": WiSSTixx,

        "WiPVeCC31":WiPVexx,    "WiPVeCS32": WiPVexx,    "WiPVeCC32": WiPVexx,   "WiPVeCT32": WiPVexx,   "WiPVeCS33": WiPVexx,   "WiPVeCC33": WiPVexx,   "WiPVePT33": WiPVexx,   "WiPVeCT34": WiPVexx,
        "WiPViSBC30": -0.8,  "WiPViENGC30": -1.0,  "WiPViPV31": WiPViPVxx,   "WiPViSST31": WiPViSSTxx,  "WiPViVIP31": WiPViVIPxx,  "WiPViPV32": WiPViPVxx,   "WiPViSST32": WiPViSSTxx,
        "WiPViPV33": WiPViPVxx,   "WiPViSST33": WiPViSSTxx,  "WiPViPV34": WiPViPVxx,  "WiPViSST34": WiPViSSTxx,

        "WiSSTeCC31":WiSSTexx,   "WiSSTeCS32": WiSSTexx,   "WiSSTeCC32": WiSSTexx,  "WiSSTeCT32": WiSSTexx,  "WiSSTeCS33": WiSSTexx,  "WiSSTeCC33": WiSSTexx,  "WiSSTePT33": WiSSTexx,  "WiSSTeCT34": WiSSTexx,
        "WiSSTiSBC30": -0.8, "WiSSTiENGC30": -1.0, "WiSSTiPV31": WiSSTixx,  "WiSSTiSST31": WiSSTixx, "WiSSTiVIP31": -1.0, "WiSSTiPV32": WiSSTixx,  "WiSSTiSST32": WiSSTixx,
        "WiSSTiPV33": WiSSTixx,  "WiSSTiSST33": WiSSTixx, "WiSSTiPV34": WiSSTixx, "WiSSTiSST34": WiSSTixx,

        "WiPVeCC41":WiPVexx,    "WiPVeCS42": WiPVexx,    "WiPVeCC42": WiPVexx,   "WiPVeCT42": WiPVexx,   "WiPVeCS43": WiPVexx,   "WiPVeCC43": WiPVexx,   "WiPVePT43": WiPVexx,   "WiPVeCT44": WiPVexx,
        "WiPViSBC40": -0.8,  "WiPViENGC40": -1.0,  "WiPViPV41": WiPViPVxx,   "WiPViSST41": WiPViSSTxx,  "WiPViVIP41": WiPViVIPxx,  "WiPViPV42": WiPViPVxx,   "WiPViSST42": WiPViSSTxx,
        "WiPViPV43": WiPViPVxx,   "WiPViSST43": WiPViSSTxx,  "WiPViPV44": WiPViPVxx,  "WiPViSST44": WiPViSSTxx,

        "WiSSTeCC41":WiSSTexx,   "WiSSTeCS42": WiSSTexx,   "WiSSTeCC42": WiSSTexx,  "WiSSTeCT42": WiSSTexx,  "WiSSTeCS43": WiSSTexx,  "WiSSTeCC43": WiSSTexx,  "WiSSTePT43": WiSSTexx,  "WiSSTeCT44": WiSSTexx,
        "WiSSTiSBC40": -0.8, "WiSSTiENGC40": -1.0, "WiSSTiPV41": WiSSTixx,  "WiSSTiSST41": WiSSTixx, "WiSSTiVIP41": -1.0, "WiSSTiPV42": WiSSTixx,  "WiSSTiSST42": WiSSTixx,
        "WiSSTiPV43": WiSSTixx,  "WiSSTiSST43": WiSSTixx, "WiSSTiPV44": WiSSTixx, "WiSSTiSST44": WiSSTixx,
        },
    'Delay':{
        "DeCCeCC11":  De, "DeCCeCS12": De,  "DeCCeCC12": De,  "DeCCeCT12": De,
        "DeCCeCS13": De,  "DeCCeCC13": De,  "DeCCePT13": De,  "DeCCeCT14": De,
        "DeCCiSBC10": De, "DeCCiENGC10": De,"DeCCiPV11": De,  "DeCCiSST11": De,"DeCCiVIP11": De,"DeCCiPV12": De,"DeCCiSST12": De,"DeCCiPV13": De,"DeCCiSST13": De,"DeCCiPV14": De,"DeCCiSST14": De,

        "DeCSeCC21": De,  "DeCSeCS22": De,  "DeCSeCC22": De,  "DeCSeCT22": De,
        "DeCSeCS23": De,  "DeCSeCC23": De,  "DeCSePT23": De,  "DeCSeCT24": De,
        "DeCSiSBC20": De, "DeCSiENGC20": De,"DeCSiPV21": De,  "DeCSiSST21": De,"DeCSiVIP21": De,"DeCSiPV22": De,"DeCSiSST22": De,"DeCSiPV23": De,"DeCSiSST23": De,"DeCSiPV24": De,"DeCSiSST24": De,

        "DeCCeCC21":De,   "DeCCeCS22": De,  "DeCCeCC22": De,  "DeCCeCT22": De, "DeCCeCS23": De, "DeCCeCC23": De,"DeCCePT23": De, "DeCCeCT24": De,
        "DeCCiSBC20": De, "DeCCiENGC20": De,"DeCCiPV21": De,  "DeCCiSST21": De,"DeCCiVIP21": De,"DeCCiPV22": De,"DeCCiSST22": De,"DeCCiPV23": De,"DeCCiSST23": De,"DeCCiPV24": De,"DeCCiSST24": De,

        "DeCTeCC21":De,   "DeCTeCS22": De,  "DeCTeCC22": De,  "DeCTeCT22": De, "DeCTeCS23": De, "DeCTeCC23": De,"DeCTePT23": De, "DeCTeCT24": De,
        "DeCTiSBC20": De, "DeCTiENGC20": De,"DeCTiPV21": De,  "DeCTiSST21": De,"DeCTiVIP21": De,"DeCTiPV22": De,"DeCTiSST22": De,"DeCTiPV23": De,"DeCTiSST23": De,"DeCTiPV24": De,"DeCTiSST24": De,

        "DeCSeCC31": De,  "DeCSeCS32": De,  "DeCSeCC32":De,  "DeCSeCT32":De,
        "DeCSeCS33": De,  "DeCSeCC33": De,  "DeCSePT33": De, "DeCSeCT34": De,
        "DeCSiSBC30": De, "DeCSiENGC30": De,"DeCSiPV31": De, "DeCSiSST31": De, "DeCSiVIP31": De,"DeCSiPV32": De,"DeCSiSST32": De,"DeCSiPV33": De,"DeCSiSST33": De,"DeCSiPV34": De,"DeCSiSST34": De,

        "DeCCeCC31":De,   "DeCCeCS32": De,  "DeCCeCC32": De, "DeCCeCT32": De,  "DeCCeCS33": De, "DeCCeCC33": De,"DeCCePT33": De, "DeCCeCT34": De,
        "DeCCiSBC30": De, "DeCCiENGC30": De,"DeCCiPV31": De, "DeCCiSST31": De, "DeCCiVIP31": De,"DeCCiPV32": De,"DeCCiSST32": De,"DeCCiPV33": De,"DeCCiSST33": De,"DeCCiPV34": De,"DeCCiSST34": De,

        "DePTeCC31": De,  "DePTeCS32": De, "DePTeCC32": De, "DePTeCT32":De,   "DePTeCS33": De, "DePTeCC33": De,"DePTePT33": De, "DePTeCT34": De,
        "DePTiSBC30": De, "DePTiENGC30": De,"DePTiPV31": De, "DePTiSST31": De, "DePTiVIP31": De,"DePTiPV32": De,"DePTiSST32": De,"DePTiPV33": De,"DePTiSST33": De,"DePTiPV34": De,"DePTiSST34": De,

        "DeCTeCC41": De,  "DeCTeCS42": De,  "DeCTeCC42": De, "DeCTeCT42": De,
        "DeCTeCS43": De,  "DeCTeCC43": De,  "DeCTePT43": De, "DeCTeCT44": De,
        "DeCTiSBC40": De, "DeCTiENGC40": De,"DeCTiPV41": De, "DeCTiSST41": De, "DeCTiVIP41": De,"DeCTiPV42": De,"DeCTiSST42": De,"DeCTiPV43": De,"DeCTiSST43": De,"DeCTiPV44": De,"DeCTiSST44": De,


        "DiSBCeCC01": De,  "DiSBCeCS02": De,  "DiSBCeCC02": De, "DiSBCeCT02": De,  "DiSBCeCS03": De,  "DiSBCeCC03": De,  "DiSBCePT03": De,  "DiSBCeCT04": De,
        "DiSBCiSBC00": De, "DiSBCiENGC00": De, "DiSBCiPV01": De,  "DiSBCiSST01": De,"DiSBCiVIP01": De, "DiSBCiPV02": De,
        "DiSBCiSST02": De, "DiSBCiPV03": De,    "DiSBCiSST03": De, "DiSBCiPV04": De,  "DiSBCiSST04": De,

        "DiENGCeCC01": De, "DiENGCeCS02": De, "DiENGCeCC02": De, "DiENGCeCT02": De, "DiENGCeCS03": De, "DiENGCeCC03": De, "DiENGCePT03": De, "DiENGCeCT04": De,
        "DiENGCiSBC00": De,"DiENGCiENGC00": De,"DiENGCiPV01": De, "DiENGCiSST01": De,"DiENGCiVIP01": De,"DiENGCiPV02": De, "DiENGCiSST02": De,"DiENGCiPV03": De, "DiENGCiSST03": De,"DiENGCiPV04": De,"DiENGCiSST04": De,

        "DiPVeCC11": De,  "DiPVeCS12": De,  "DiPVeCC12": De, "DiPVeCT12": De,  "DiPVeCS13": De, "DiPVeCC13": De, "DiPVePT13": De, "DiPVeCT14": De,
        "DiPViSBC10": De, "DiPViENGC10": De, "DiPViPV11": De, "DiPViSST11": De, "DiPViVIP11": De, "DiPViPV12": De, "DiPViSST12": De, "DiPViPV13": De, "DiPViSST13": De,  "DiPViPV14": De, "DiPViSST14": De,

        "DiSSTeCC11":De,  "DiSSTeCS12": De, "DiSSTeCC12": De,  "DiSSTeCT12": De,  "DiSSTeCS13": De,  "DiSSTeCC13": De,  "DiSSTePT13": De,  "DiSSTeCT14": De,
        "DiSSTiSBC10": De, "DiSSTiENGC10": De, "DiSSTiPV11": De, "DiSSTiSST11": De, "DiSSTiVIP11": De, "DiSSTiPV12": De, "DiSSTiSST12": De, "DiSSTiPV13": De, "DiSSTiSST13": De, "DiSSTiPV14": De, "DiSSTiSST14": De,

        "DiVIPeCC11": De,   "DiVIPeCS12": De,   "DiVIPeCC12": De,  "DiVIPeCT12": De,  "DiVIPeCS13": De,  "DiVIPeCC13": De,  "DiVIPePT13": De,  "DiVIPeCT14": De,
        "DiVIPiSBC10": De, "DiVIPiENGC10": De, "DiVIPiPV11": De,  "DiVIPiSST11": De, "DiVIPiVIP11": De, "DiVIPiPV12": De,  "DiVIPiSST12": De, "DiVIPiPV13": De,  "DiVIPiSST13": De, "DiVIPiPV14": De, "DiVIPiSST14": De,

        "DiPVeCC21": De,    "DiPVeCS22": De,  "DiPVeCC22": De, "DiPVeCT22": De,   "DiPVeCS23": De,  "DiPVeCC23": De, "DiPVePT23": De, "DiPVeCT24": De,
        "DiPViSBC20": De,  "DiPViENGC20": De, "DiPViPV21": De, "DiPViSST21": De, "DiPViVIP21": De,  "DiPViPV22": De,   "DiPViSST22": De, "DiPViPV23": De, "DiPViSST23": De, "DiPViPV24": De, "DiPViSST24": De,

        "DiSSTeCC21": De, "DiSSTeCS22": De,   "DiSSTeCC22": De,  "DiSSTeCT22": De,  "DiSSTeCS23": De,  "DiSSTeCC23": De,  "DiSSTePT23": De,  "DiSSTeCT24": De,
        "DiSSTiSBC20": De, "DiSSTiENGC20": De, "DiSSTiPV21": De,  "DiSSTiSST21": De, "DiSSTiVIP21": De, "DiSSTiPV22": De, "DiSSTiSST22": De, "DiSSTiPV23": De, "DiSSTiSST23": De, "DiSSTiPV24": De, "DiSSTiSST24": De,

        "DiPVeCC31": De,    "DiPVeCS32": De,    "DiPVeCC32": De,   "DiPVeCT32": De,   "DiPVeCS33": De,   "DiPVeCC33": De,   "DiPVePT33": De,   "DiPVeCT34": De,
        "DiPViSBC30": De,  "DiPViENGC30": De,  "DiPViPV31": De,  "DiPViSST31": De, "DiPViVIP31": De, "DiPViPV32": De, "DiPViSST32": De,  "DiPViPV33": De, "DiPViSST33": De, "DiPViPV34": De, "DiPViSST34": De,

        "DiSSTeCC31": De,   "DiSSTeCS32": De,   "DiSSTeCC32": De,  "DiSSTeCT32": De,  "DiSSTeCS33": De,  "DiSSTeCC33": De,  "DiSSTePT33": De,  "DiSSTeCT34": De,
        "DiSSTiSBC30": De, "DiSSTiENGC30": De, "DiSSTiPV31": De, "DiSSTiSST31": De, "DiSSTiVIP31": De, "DiSSTiPV32": De,  "DiSSTiSST32": De, "DiSSTiPV33": De,  "DiSSTiSST33": De, "DiSSTiPV34": De, "DiSSTiSST34": De,

        "DiPVeCC41": De,  "DiPVeCS42": De,  "DiPVeCC42": De, "DiPVeCT42": De, "DiPVeCS43": De, "DiPVeCC43": De, "DiPVePT43": De, "DiPVeCT44": De,
        "DiPViSBC40": De,  "DiPViENGC40": De, "DiPViPV41": De, "DiPViSST41": De, "DiPViVIP41": De, "DiPViPV42": De, "DiPViSST42": De, "DiPViPV43": De, "DiPViSST43": De, "DiPViPV44": De, "DiPViSST44": De,

        "DiSSTeCC41": De, "DiSSTeCS42": De, "DiSSTeCC42": De, "DiSSTeCT42": De,  "DiSSTeCS43": De,  "DiSSTeCC43": De,  "DiSSTePT43": De,  "DiSSTeCT44": De,
        "DiSSTiSBC40": De, "DiSSTiENGC40": De, "DiSSTiPV41": De, "DiSSTiSST41": De, "DiSSTiVIP41": De, "DiSSTiPV42": De,  "DiSSTiSST42": De, "DiSSTiPV43": De, "DiSSTiSST43": De, "DiSSTiPV44": De, "DiSSTiSST44": De,
        },
}

# making the 3d matrix for the connection matrix was the idea that I did not continue it... later I never used this part
import numpy as np

E_cells = ['CC','CS', 'CC', 'CT', 'CS', 'CC', 'PT', 'CT']
I_cells = ['SBC' ,'ENGC' , 'PV' , 'SST', 'VIP', 'PV', 'SST', 'PV', 'SST', 'PV', 'SST']
EI_cells = E_cells + I_cells
E_count = ['1','2','2','2','3','3','3','4']
I_count = ['0','0','1','1','1','2','2','3','3','4','4']
EI_count = E_count + I_count
parameters= ['Probability','Sigma','Weight', 'Delay']
M1_connection_matrix=np.zeros((4,19,19))
for l in range(len(M1_connection_matrix)):
    for j in range(len(M1_connection_matrix[0])):
        for i in range(len(M1_connection_matrix[0])):
            m='e' if j<8 else 'i'
            n='e' if i<8 else 'i'
            M1_connection_matrix[l][j][i]=connect_matrix[parameters[l]][parameters[l][0]+ m +EI_cells[j]+ n +EI_cells[i]+EI_count[j]+EI_count[i]]  #       PeCCeCC11'']

#######################################################################################################################
# creation of dictionary file for reading the connection information of connection of layers
# this is kind of substitution for the big matrix 'M1_connection_matrix' I made.
#######################################################################################################################
# the dictionary name is "M1_internal_dict". I will use it later to make 'pickle' file
scale_ls_e = range(8)
scale_ls_i = range(11)
index_ls_e = ['M1_L23_CC','M1_L5A_CS', 'M1_L5A_CC', 'M1_L5A_CT', 'M1_L5B_CS', 'M1_L5B_CC', 'M1_L5B_PT', 'M1_L6_CT']
index_ls_i = ['M1_L1_SBC' ,'M1_L1_ENGC' , 'M1_L23_PV' , 'M1_L23_SST', 'M1_L23_VIP', 'M1_L5A_PV', 'M1_L5A_SST', 'M1_L5B_PV', 'M1_L5B_SST', 'M1_L6_PV', 'M1_L6_SST']

#index_ls_e = ['L23_CC','L5A_CS', 'L5A_CC', 'L5A_CT', 'L5B_CS', 'L5B_CC', 'L5B_PT', 'L6_CT']
#index_ls_i = ['L1_SBC' ,'L1_ENGC' , 'L23_PV' , 'L23_SST', 'L23_VIP', 'L5A_PV', 'L5A_SST', 'L5B_PV', 'L5B_SST', 'L6_PV', 'L6_SST']

from collections import defaultdict
M1_internal_dict = defaultdict(dict)
for i in scale_ls_e:
    for j in scale_ls_e:
        M1_internal_dict[index_ls_e[i]][index_ls_e[j]] = {
                        'p_center': connect_matrix[parameters[0]][parameters[0][0]+ 'e' +E_cells[j]+ 'e' +E_cells[i]+E_count[j]+E_count[i]],
                        'sigma': connect_matrix[parameters[1]][parameters[1][0]+ 'e' +E_cells[j]+ 'e' +E_cells[i]+E_count[j]+E_count[i]],
                        'weight': connect_matrix[parameters[2]][parameters[2][0]+ 'e' +E_cells[j]+ 'e' +E_cells[i]+E_count[j]+E_count[i]],
                        'delay': connect_matrix[parameters[3]][parameters[3][0]+ 'e' +E_cells[j]+ 'e' +E_cells[i]+E_count[j]+E_count[i]],
                        'weight_distribution': 'lognormal'}

for i in scale_ls_e:
    for j in scale_ls_i:
        M1_internal_dict[index_ls_e[i]][index_ls_i[j]] = {
                        'p_center': connect_matrix[parameters[0]][parameters[0][0]+ 'i' +I_cells[j]+ 'e' +E_cells[i]+I_count[j]+E_count[i]],
                        'sigma': connect_matrix[parameters[1]][parameters[1][0]+ 'i' +I_cells[j]+ 'e' +E_cells[i]+I_count[j]+E_count[i]],
                        'weight': connect_matrix[parameters[2]][parameters[2][0]+ 'i' +I_cells[j]+ 'e' +E_cells[i]+I_count[j]+E_count[i]],
                        'delay': connect_matrix[parameters[3]][parameters[3][0]+ 'i' +I_cells[j]+ 'e' +E_cells[i]+I_count[j]+E_count[i]],
                        'weight_distribution': 'homogeneous'}

for i in scale_ls_i:
    for j in scale_ls_e:
        M1_internal_dict[index_ls_i[i]][index_ls_e[j]] = {
                        'p_center': connect_matrix[parameters[0]][parameters[0][0]+ 'e' +E_cells[j]+ 'i' +I_cells[i]+E_count[j]+I_count[i]],
                        'sigma': connect_matrix[parameters[1]][parameters[1][0]+ 'e' +E_cells[j]+ 'i' +I_cells[i]+E_count[j]+I_count[i]],
                        'weight': connect_matrix[parameters[2]][parameters[2][0]+ 'e' +E_cells[j]+ 'i' +I_cells[i]+E_count[j]+I_count[i]],
                        'delay': connect_matrix[parameters[3]][parameters[3][0]+ 'e' +E_cells[j]+ 'i' +I_cells[i]+E_count[j]+I_count[i]],
                        'weight_distribution': 'homogeneous'}

for i in scale_ls_i:
    for j in scale_ls_i:
        M1_internal_dict[index_ls_i[i]][index_ls_i[j]] = {
                        'p_center': connect_matrix[parameters[0]][parameters[0][0]+ 'i' +I_cells[j]+ 'i' +I_cells[i]+I_count[j]+I_count[i]],
                        'sigma': connect_matrix[parameters[1]][parameters[1][0]+ 'i' +I_cells[j]+ 'i' +I_cells[i]+I_count[j]+I_count[i]],
                        'weight': connect_matrix[parameters[2]][parameters[2][0]+ 'i' +I_cells[j]+ 'i' +I_cells[i]+I_count[j]+I_count[i]],
                        'delay': connect_matrix[parameters[3]][parameters[3][0]+ 'i' +I_cells[j]+ 'i' +I_cells[i]+I_count[j]+I_count[i]],
                        'weight_distribution': 'homogeneous'}



#######################################################################################################################
import pickle

with open('M1_internal_connection.pickle', 'wb') as handle:
    pickle.dump(M1_internal_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

########################################################################################################################
# Factor for the connection probability of postsynaptic excitatory cells from presynaptic excitatory cells
# Given that maximum connection probability was 24.3 % within layer 4 in
# somatosensory cortex by Lefort et al., 2009, 30 % was set as starting point
# as maximamu connection probability.
########################################################################################################################


# Base parameters and their meaning
# these defaults can be overrided via a custom python parameter file or via the commandline (in this order: commandline arguments take precedence over
#  customParams.py, which take precendence over the defaults defined here)

####################################################################################
# This file should not be modified! Use commandline arguments or custom param file #
####################################################################################
ctxM1Params={
        'M1':{
            'structure_info':{
            'region_name': "M1",
            'region_size':[1., 1.0, 1.4],
            'layer_thickness':[0.14, 0.304, 0.2, 0.471, 0.285],
            'layer_depth':[0., 0.14, 0.444, 0.644, 1.115, 1.4],
            'Layer_Name':['L1', 'L23', 'L5A', 'L5B', 'L6'],
            'Layer_Cellcount_mm2': [1799.99,18320.334,6395.5168,15182.9,17628.]     # Cell density is set based on /home/morteza/Desktop/Research/Brain lab/brain circuits/connectivity matrix/connectivity
            },

            'neuro_info':{

                'L1':{
                    'SBC':{
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 1259.1928,
                        "n_type_index" :0
                    },
                    'ENGC':{
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 540.8071,
                        "n_type_index" :1
                    }
                },

                'L23': {
                    'CC': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "E",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 14659.2,
                        "n_type_index": 0
                    },
                    'PV': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 1220.3784,
                        "n_type_index": 1
                    },
                    'SST': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 1220.3784,
                        "n_type_index": 2
                    },
                    'VIP': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 1220.3784,
                        "n_type_index": 3
                    },
                },

                'L5A': {
                    'CS': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "E",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 1702.0327,
                        "n_type_index": 0
                   },
                    'CC': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "E",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 1702.0327,
                        "n_type_index": 1
                    },
                    'CT': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "E",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 1702.0327,
                        "n_type_index": 2
                   },
                    'PV': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 773.6512,
                        "n_type_index": 3
                    },
                    'SST': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 515.7675,
                        "n_type_index": 4
                    },
                },

                'L5B': {
                    'PT': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "E",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 6073.1624,
                        "n_type_index": 0
                    },
                   'CS': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "E",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 3036.5812,
                        "n_type_index": 1
                   },
                    'CC': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "E",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 3036.5812,
                        "n_type_index": 2
                    },
                    'PV': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 1821.9487,
                        "n_type_index": 3
                    },
                    'SST': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 1214.6324,
                        "n_type_index": 4
                   },
                },

                'L6': {
                    'CT': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "E",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 14102.4,
                        "n_type_index": 0
                    },
                    'PV': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 1762.8,
                        "n_type_index": 1
                   },
                    'SST': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 1762.8,
                        "n_type_index": 2
                   },
                },
            },
                'connection_info':{
                'M1toM1':'M1_internal_connection.pickle'
                },

                'position_type_info': {
                    'L1':'M1_Neuron_pos_L1.npz',
                    'L23':'M1_Neuron_pos_L23.npz',
                    'L5A':'M1_Neuron_pos_L5A.npz',
                    'L5B':'M1_Neuron_pos_L5B.npz',
                    'L6':'M1_Neuron_pos_L6.npz'
                },

            }
        }







