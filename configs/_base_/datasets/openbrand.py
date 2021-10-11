dataset_type = 'OpenBrandDataset'
data_root = '/home/ubuntu/data/OpenBrands/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

policies = [
    [
        dict(type='Posterize', bits=4, prob=0.4),
        dict(type='Rotate', angle=30., prob=0.6)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
        dict(type='AutoContrast', prob=0.6)
    ],
    [dict(type='Equalize', prob=0.8),
     dict(type='Equalize', prob=0.6)],
    [
        dict(type='Posterize', bits=5, prob=0.6),
        dict(type='Posterize', bits=5, prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 6, prob=0.6),
        dict(type='Equalize', prob=0.6)
    ],
    [dict(type='Posterize', bits=6, prob=0.8),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='Rotate', angle=10., prob=0.2),
        dict(type='Solarize', thr=256 / 9, prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.6),
        dict(type='Posterize', bits=5, prob=0.4)
    ],
    [
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8),
        dict(type='ColorTransform', magnitude=0., prob=0.4)
    ],
    [
        dict(type='Rotate', angle=30., prob=0.4),
        dict(type='Equalize', prob=0.6)
    ],
    [dict(type='Equalize', prob=0.0),
     dict(type='Equalize', prob=0.8)],
    [dict(type='Invert', prob=0.6),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='ColorTransform', magnitude=0.4, prob=0.6),
        dict(type='Contrast', magnitude=0.8, prob=1.)
    ],
    [
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8),
        dict(type='ColorTransform', magnitude=0.2, prob=1.)
    ],
    [
        dict(type='ColorTransform', magnitude=0.8, prob=0.8),
        dict(type='Solarize', thr=256 / 9 * 2, prob=0.8)
    ],
    [
        dict(type='Sharpness', magnitude=0.7, prob=0.4),
        dict(type='Invert', prob=0.6)
    ],
    [
        dict(
            type='Shear',
            magnitude=0.3 / 9 * 5,
            prob=0.6,
            direction='horizontal'),
        dict(type='Equalize', prob=1.)
    ],
    [
        dict(type='ColorTransform', magnitude=0., prob=0.4),
        dict(type='Equalize', prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
        dict(type='AutoContrast', prob=0.6)
    ],
    [dict(type='Invert', prob=0.6),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='ColorTransform', magnitude=0.4, prob=0.6),
        dict(type='Contrast', magnitude=0.8, prob=1.)
    ],
    [dict(type='Equalize', prob=0.8),
     dict(type='Equalize', prob=0.6)],
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CropBoundingBox'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='AutoAugment', policies=policies),
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        mode='const',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='SwapChannels'),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CropBoundingBox'),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='SwapChannels'),
    dict(type='Collect', keys=['img']),
]

openbrand_classes = ["LAMY", "tumi", "warrior", "sandisk", "belle", "ThinkPad", "rolex", "balabala", "vlone", "nanfu", "KTM", "VW", "libai", "snoopy", "Budweiser", "armani", "gree", "GOON", "KielJamesPatrick", "uniqlo", "peppapig", "valentino", "GUND", "christianlouboutin", "toyota", "moutai", "semir", "marcjacobs", "esteelauder", "chaoneng", "goldsgym", "airjordan", "bally", "fsa", "jaegerlecoultre", "dior", "samsung", "fila", "hellokitty", "Jansport", "barbie", "VDL", "manchesterunited", "coach", "PopSockets", "haier", "banbao", "omron", "fendi", "erke", "lachapelle", "chromehearts", "leader", "pantene", "motorhead", "girdear", "fresh", "katespade", "pandora", "Aape", "edwin", "yonghui", "Levistag", "kboxing", "yili", "ugg", "CommedesGarcons", "Bosch", "palmangels", "razer", "guerlain", "balenciaga", "anta", "Duke", "kingston", "nestle", "FGN", "vrbox", "toryburch", "teenagemutantninjaturtles", "converse", "nanjiren", "Josiny", "kappa", "nanoblock", "lincoln", "michael_kors", "skyworth", "olay", "cocacola", "swarovski", "joeone", "lining", "joyong", "tudor", "YEARCON", "hyundai", "OPPO", "ralphlauren", "keds", "amass", "thenorthface", "qingyang", "mujosh", "baishiwul", "dissona", "honda", "newera", "brabus", "hera", "titoni", "decathlon", "DanielWellington", "moony", "etam", "liquidpalisade", "zippo", "mistine", "eland", "wodemeiliriji", "ecco", "xtep", "piaget", "gloria", "hp", "loewe", "Levis_AE", "Anna_sui", "MURATA", "durex", "zebra", "kanahei", "ihengima", "basichouse", "hla", "ochirly", "chloe", "miumiu", "aokang", "SUPERME", "simon", "bosideng", "brioni", "moschino", "jimmychoo", "adidas", "lanyueliang", "aux", "furla", "parker", "wechat", "emiliopucci", "bmw", "monsterenergy", "Montblanc", "castrol", "HUGGIES", "bull", "zhoudafu", "leaders", "tata", "oldnavy", "OTC", "levis", "veromoda", "Jmsolution", "triangle", "Specialized", "tries", "pinarello", "Aquabeads", "deli", "mentholatum", "molsion", "tiffany", "moco", "SANDVIK", "franckmuller", "oakley", "bulgari", "montblanc", "beaba", "nba", "shelian", "puma", "PawPatrol", "offwhite", "baishiwuliu", "lexus", "cainiaoguoguo", "hugoboss", "FivePlus", "shiseido", "abercrombiefitch", "rejoice", "mac", "chigo", "pepsicola", "versacetag", "nikon", "TOUS", "huawei", "chowtaiseng", "Amii", "jnby", "jackjones", "THINKINGPUTTY", "bose", "xiaomi", "moussy", "Miss_sixty", "Stussy", "stanley", "loreal", "dhc", "sulwhasoo", "gentlemonster", "midea", "beijingweishi", "mlb", "cree", "dove", "PJmasks", "reddragonfly", "emerson", "lovemoschino", "suzuki", "erdos", "seiko", "cpb", "royalstar", "thehistoryofwhoo", "otterbox", "disney", "lindafarrow", "PATAGONIA", "seven7", "ford", "bandai", "newbalance", "alibaba", "sergiorossi", "lacoste", "bear", "opple", "walmart", "clinique", "asus", "ThomasFriends", "wanda", "lenovo", "metallica", "stuartweitzman", "karenwalker", "celine", "miui", "montagut", "pampers", "darlie", "toray", "bobdog", "ck", "flyco", "alexandermcqueen", "shaxuan", "prada", "miiow", "inman", "3t", "gap", "Yamaha", "fjallraven", "vancleefarpels", "acne", "audi", "hunanweishi", "henkel", "mg", "sony", "CHAMPION", "iwc", "lv", "dolcegabbana", "avene", "longchamp", "anessa", "satchi", "hotwheels", "nike", "hermes", "jiaodan", "siemens", "Goodbaby", "innisfree", "Thrasher", "kans", "kenzo", "juicycouture", "evisu", "volcom", "CanadaGoose", "Dickies", "angrybirds", "eddrac", "asics", "doraemon", "hisense", "juzui", "samsonite", "hikvision", "naturerepublic", "Herschel", "MANGO", "diesel", "hotwind", "intel", "arsenal", "rayban", "tommyhilfiger", "ELLE", "stdupont", "ports", "KOHLER", "thombrowne", "mobil", "Belif", "anello", "zhoushengsheng", "d_wolves", "FridaKahlo", "citizen", "fortnite", "beautyBlender", "alexanderwang", "charles_keith", "panerai", "lux", "beats", "Y-3", "mansurgavriel", "goyard", "eral", "OralB", "markfairwhale", "burberry", "uno", "okamoto", "only", "bvlgari", "heronpreston", "jimmythebull", "dyson", "kipling", "jeanrichard", "PXG", "pinkfong", "Versace", "CCTV", "paulfrank", "lanvin", "vans", "cdgplay", "baojianshipin", "rapha", "tissot", "casio", "patekphilippe", "tsingtao", "guess", "Lululemon", "hollister", "dell", "supor", "MaxMara", "metersbonwe", "jeanswest", "lancome", "lee", "omega", "lets_slim", "snp", "PINKFLOYD", "cartier", "zenith", "LG", "monchichi", "hublot", "benz", "apple", "blackberry", "wuliangye", "porsche", "bottegaveneta", "instantlyageless", "christopher_kane", "bolon", "tencent", "dkny", "aptamil", "makeupforever", "kobelco", "meizu", "vivo", "buick", "tesla", "septwolves", "samanthathavasa", "tomford", "jeep", "canon", "nfl", "kiehls", "pigeon", "zhejiangweishi", "snidel", "hengyuanxiang", "linshimuye", "toread", "esprit", "BASF", "gillette", "361du", "bioderma", "UnderArmour", "TommyHilfiger", "ysl", "onitsukatiger", "house_of_hello", "baidu", "robam", "konka", "jack_wolfskin", "office", "goldlion", "tiantainwuliu", "wonderflower", "arcteryx", "threesquirrels", "lego", "mindbridge", "emblem", "grumpycat", "bejirog", "ccdd", "3concepteyes", "ferragamo", "thermos", "Auby", "ahc", "panasonic", "vanguard", "FESTO", "MCM", "lamborghini", "laneige", "ny", "givenchy", "zara", "jiangshuweishi", "daphne", "longines", "camel", "philips", "nxp", "skf", "perfect", "toshiba", "wodemeilirizhi", "Mexican", "VANCLEEFARPELS", "HARRYPOTTER", "mcm", "nipponpaint", "chenguang", "jissbon", "versace", "girardperregaux", "chaumet", "columbia", "nissan", "3M", "yuantong", "sk2", "liangpinpuzi", "headshoulder", "youngor", "teenieweenie", "tagheuer", "starbucks", "pierrecardin", "vacheronconstantin", "peskoe", "playboy", "chanel", "HarleyDavidson_AE", "volvo", "be_cheery", "mulberry", "musenlin", "miffy", "peacebird", "tcl", "ironmaiden", "skechers", "moncler", "rimowa", "safeguard", "baleno", "sum37", "holikaholika", "gucci", "theexpendables", "dazzle", "vatti", "nintendo"]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_1_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_1/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_2_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_2/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_3_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_3/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_4_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_4/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_5_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_5/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_6_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_6/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_7_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_7/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_8_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_8/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_9_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_9/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_10_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_10/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_11_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_11/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_12_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_12/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_13_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_13/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_14_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_14/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_15_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_15/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_16_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_16/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_17_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_17/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_18_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_18/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_19_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_19/',
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/train_20210409_20_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_20/',
                pipeline=train_pipeline,
            ),
        ],
    ),
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_1_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_1/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_2_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_2/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_3_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_3/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_4_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_4/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_5_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_5/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_6_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_6/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_7_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_7/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_8_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_8/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_9_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_9/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_10_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_10/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_11_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_11/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_12_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_12/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_13_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_13/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_14_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_14/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_15_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_15/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_16_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_16/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_17_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_17/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_18_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_18/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_19_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_19/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/validation_20210409_20_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_20/',
                pipeline=test_pipeline,
            ),
        ],
    ),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_1_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_1/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_2_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_2/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_3_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_3/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_4_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_4/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_5_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_5/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_6_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_6/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_7_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_7/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_8_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_8/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_9_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_9/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_10_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_10/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_11_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_11/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_12_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_12/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_13_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_13/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_14_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_14/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_15_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_15/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_16_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_16/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_17_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_17/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_18_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_18/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_19_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_19/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                ann_file=data_root + 'annotations/test_20210409_20_reduced.json',
                data_prefix=data_root + '电商标识检测大赛_train_20210409_20/',
                pipeline=test_pipeline,
            ),
        ],
    ),
)