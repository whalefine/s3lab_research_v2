from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.dtb70_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/dtb70/DTB70'
    settings.got10k_lmdb_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/got10k_lmdb'
    settings.got10k_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/itb'
    settings.lasot_extension_subset_path_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/lasot_lmdb'
    settings.lasot_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/lasot'
    settings.network_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/nfs'
    settings.otb_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/otb'
    settings.prj_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python'
    settings.result_plot_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/output/test/result_plots'
    settings.results_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/output'
    settings.segmentation_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/output/test/segmentation_results'
    settings.tc128_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/trackingnet'
    settings.uav123_10fps_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/uav123_10fps/UAV123_10fps'
    settings.uav123_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/uav123/UAV123'
    settings.uav_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/uav'
    settings.uavdt_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/uavdt/home/data/uavdt'
    settings.uavtrack_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/uavtrack112/home/data/V4RFlight112'
    settings.visdrone_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/visdrone/VisDrone2018-SOT-test-dev'
    settings.vot18_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/vot2018'
    settings.vot22_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/vot2022'
    settings.vot_path = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

