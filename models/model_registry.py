from .maddpg import MADDPG
from .sqddpg import SQDDPG
from .iac import IAC
from .iddpg import IDDPG
from .coma import COMA
from .maac import MAAC
from .matd3 import MATD3
from .matd3_reference import MATD3Reference
from .ippo import IPPO
from .mappo import MAPPO
from .facmaddpg import FACMADDPG



Model = dict(maddpg=MADDPG,
             sqddpg=SQDDPG,
             iac=IAC,
             iddpg=IDDPG,
             coma=COMA,
             maac=MAAC,
             matd3=MATD3,
             matd3_legacy=MATD3,
             matd3_reference=MATD3Reference,
             ippo=IPPO,
             mappo=MAPPO,
             facmaddpg=FACMADDPG
            )

Strategy = dict(maddpg='pg',
                sqddpg='pg',
                iac='pg',
                iddpg='pg',
                coma='pg',
                maac='pg',
                matd3='pg',
                matd3_legacy='pg',
                matd3_reference='pg',
                ippo='pg',
                mappo='pg',
                facmaddpg='pg'
            )
