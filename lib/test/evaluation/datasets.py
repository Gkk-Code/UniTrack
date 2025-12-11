from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    otb=DatasetInfo(module=pt % "otb", class_name="OTBDataset", kwargs=dict()),
    nfs=DatasetInfo(module=pt % "nfs", class_name="NFSDataset", kwargs=dict()),
    uav=DatasetInfo(module=pt % "uav", class_name="UAVDataset", kwargs=dict()),
    tc128=DatasetInfo(module=pt % "tc128", class_name="TC128Dataset", kwargs=dict()),
    tc128ce=DatasetInfo(module=pt % "tc128ce", class_name="TC128CEDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    lasot_lmdb=DatasetInfo(module=pt % "lasot_lmdb", class_name="LaSOTlmdbDataset", kwargs=dict()),
    lasot_extension_subset = DatasetInfo(module=pt % "lasotextensionsubset", class_name="LaSOTExtensionSubsetDataset",
                                         kwargs=dict()),
    otb99_lang=DatasetInfo(module=pt % "otb99lang", class_name="OTB99LangDataset", kwargs=dict()),
    tnl2k=DatasetInfo(module=pt % "tnl2k", class_name="TNL2kDataset", kwargs=dict()),
    lasot_lang=DatasetInfo(module=pt % "lasotlang", class_name="LaSOTLangDataset", kwargs=dict()),



    darktrack2021=DatasetInfo(module=pt % "darktrack2021", class_name="DarkTrack2021Dataset",
                              kwargs=dict()),
    uavdark135=DatasetInfo(module=pt % "uavdark135", class_name="UAVDark135Dataset",
                           kwargs=dict()),
    nat2021=DatasetInfo(module=pt % "nat2021", class_name="NAT2021Dataset",
                        kwargs=dict()),
    nat2021l=DatasetInfo(module=pt % "nat2021L", class_name="NAT2021LDataset",
                         kwargs=dict()),
    nat2024=DatasetInfo(module=pt % "nat2024", class_name="NAT2024Dataset",
                        kwargs=dict()),
    nut=DatasetInfo(module=pt % "nut", class_name="NUTDataset",
                    kwargs=dict()),

    uav123_10fps=DatasetInfo(module=pt % "uav123_10fps", class_name="UAV123_10fpsDataset", kwargs=dict()),
    uav123=DatasetInfo(module=pt % "uav123", class_name="UAV123Dataset", kwargs=dict()),
    visdrone=DatasetInfo(module=pt % "visdrone", class_name="VISDRONEDataset", kwargs=dict()),
    uavdt=DatasetInfo(module=pt % "uavdt", class_name="UAVDTDataset", kwargs=dict()),
    dtb70=DatasetInfo(module=pt % "dtb70", class_name="DTB70Dataset", kwargs=dict()),
    uavtrack=DatasetInfo(module=pt % "uavtrack", class_name="UAVTrackDataset", kwargs=dict()),
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset