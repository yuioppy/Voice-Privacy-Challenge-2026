# We need to set CUDA_VISIBLE_DEVICES before we import Pytorch, so we will read all arguments directly on startup
from argparse import ArgumentParser
import os
from pathlib import Path
parser = ArgumentParser()

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ('true', 'yes', '1')
    return bool(v)

parser.add_argument('--config', default='configs/track1/anon_mcadams.yaml')
parser.add_argument('--gpu_ids', default='0')
parser.add_argument('--force_compute', default='false', type=str)
args = parser.parse_args()
args.force_compute = _str_to_bool(args.force_compute)

if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # do not overwrite previously set devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
else: # CUDA_VISIBLE_DEVICES more important than the gpu_ids arg
    args.gpu_ids = ",".join([ str(i) for i, _ in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(","))])

config_path = Path(args.config)
if not config_path.exists():
    parser.error(
        f"Config file not found: {config_path}. "
        "Use --config with one of the YAML files in configs/track1 or configs/track2."
    )

def _get_dataset_paths(config):
    datasets = {}
    data_dir = Path(config.get('data_dir', 'data')).expanduser()
    for dataset in config['datasets']:
        no_sub = True
        for subset_name in ['trials', 'enrolls']:
            if subset_name in dataset:
                for subset in dataset[subset_name]:
                    dataset_name = f'{dataset["data"]}{subset}'
                    datasets[dataset_name] = data_dir / dataset_name
                    no_sub = False
        if no_sub:
            dataset_name = dataset["data"]
            datasets[dataset_name] = data_dir / dataset_name
    return datasets


def _validate_dataset_inputs(config, config_path):
    datasets = _get_dataset_paths(config)
    missing = [f'{name}: {path / "wav.scp"}' for name, path in datasets.items()
               if not (path / 'wav.scp').exists()]
    missing_audio = []
    for name, path in datasets.items():
        wav_scp = path / 'wav.scp'
        if not wav_scp.exists():
            continue
        checked = 0
        missing_count = 0
        first_missing = None
        with open(wav_scp, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                checked += 1
                utt, wav_ref = line.strip().split(maxsplit=1)
                if wav_ref.endswith('|'):
                    candidates = [
                        token for token in wav_ref[:-1].strip().split()
                        if Path(token).suffix.lower() in {'.wav', '.flac', '.mp3'}
                    ]
                    if not candidates:
                        continue
                    wav_ref = candidates[-1]
                if not Path(wav_ref).expanduser().exists():
                    missing_count += 1
                    if first_missing is None:
                        first_missing = f'{utt}: {wav_ref}'
        if missing_count:
            missing_audio.append(f'{name}: {missing_count}/{checked} missing audio files, first missing: {first_missing}')

    if not missing and not missing_audio:
        return

    setup_hint = (
        "Run `bash 01_download_data_model_track1.sh` first. "
        "For Track 1, IEMOCAP must also be downloaded separately and linked to `data/IEMOCAP/wav`."
        if 'track1' in config_path.parts
        else "Run `bash 01_download_data_model_track2.sh` first."
        if 'track2' in config_path.parts
        else "Prepare the configured data directories before running anonymization."
    )
    parser.error(
        "Missing required input files for configured datasets:\n  "
        + "\n  ".join(missing + missing_audio)
        + f"\n{setup_hint}"
    )


try:
    from hyperpyyaml import load_hyperpyyaml
except ModuleNotFoundError as exc:
    parser.error(f"{exc}. Install project requirements before running anonymization.")

with open(config_path, 'r') as f:
    _validate_dataset_inputs(load_hyperpyyaml(f), config_path)

import subprocess
import sys
import torch

# Apply torchaudio compatibility patch before importing speechbrain
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        return ['soundfile', 'sox', 'ffmpeg']
    torchaudio.list_audio_backends = list_audio_backends

from utils import parse_yaml, get_datasets, check_dependencies, setup_logger

logger = setup_logger(__name__)

def shell_run(cmd):
    if subprocess.run(['bash', cmd]).returncode != 0:
        logger.error(f'Failed to bash execute: {cmd}')
        sys.exit(1)


def check_dependencies_in_venv(requirements_file, venv_dir):
    """Run check_dependencies using the given venv's Python (for pipelines with isolated venv)."""
    venv_python = Path(venv_dir) / 'bin' / 'python'
    if not venv_python.exists():
        venv_python = Path(venv_dir) / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        raise FileNotFoundError(f'Python not found in venv: {venv_dir}')
    code = f"import sys; sys.path.insert(0, '.'); from utils import check_dependencies; check_dependencies('{requirements_file}')"
    result = subprocess.run([str(venv_python), '-c', code], cwd=Path(__file__).parent)
    if result.returncode != 0:
        sys.exit(result.returncode)

if __name__ == '__main__':

    config = parse_yaml(config_path)

    # For isolated-venv pipelines: run install first (creates venv), then re-exec with that venv
    if config['pipeline'] in ('sttts', 'sttts_multi'):
        install_script = 'anonymization/pipelines/sttts/install.sh' if config['pipeline'] == 'sttts' else 'anonymization/pipelines/sttts_multi/install.sh'
        shell_run(install_script)
        venv_name = 'venv_sttts' if config['pipeline'] == 'sttts' else 'venv_sttts_multi'
        venv_python = Path(__file__).parent / venv_name / 'bin' / 'python'
        if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)

    datasets = get_datasets(config)

    gpus = args.gpu_ids.split(',')

    devices = []
    if torch.cuda.is_available():
        for gpu in gpus:
            devices.append(torch.device(f'cuda:{gpu}'))
    else:
        devices.append(torch.device('cpu'))

    if config['pipeline'] == "mcadams":
        from anonymization.pipelines.mcadams import McAdamsPipeline as pipeline
    elif config['pipeline'] == "sttts":
        shell_run('anonymization/pipelines/sttts/install.sh')
        check_dependencies_in_venv('anonymization/pipelines/sttts/requirements.txt',
                                   Path(__file__).parent / 'venv_sttts')
        if "download_precomputed_intermediate_repr" in config and config["download_precomputed_intermediate_repr"]:
            shell_run('anonymization/pipelines/sttts/download_precomputed_intermediate_repr.sh')
        venv_dir = Path(__file__).parent / 'venv_sttts'
        os.environ['ESPEAK_DATA_PATH'] = str(venv_dir / 'share' / 'espeak-ng-data')
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(venv_dir / 'lib' / 'libespeak-ng.so')
        from anonymization.pipelines.sttts import STTTSPipeline as pipeline
    elif config['pipeline'] == "sttts_multi":
        shell_run('anonymization/pipelines/sttts_multi/install.sh')
        check_dependencies_in_venv('anonymization/pipelines/sttts_multi/requirements.txt',
                                   Path(__file__).parent / 'venv_sttts_multi')
        if "download_precomputed_intermediate_repr" in config and config["download_precomputed_intermediate_repr"]:
            shell_run('anonymization/pipelines/sttts_multi/download_precomputed_intermediate_repr.sh')
        venv_dir = Path(__file__).parent / 'venv_sttts_multi'
        os.environ['ESPEAK_DATA_PATH'] = str(venv_dir / 'share' / 'espeak-ng-data')
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(venv_dir / 'lib' / 'libespeak-ng.so')
        from anonymization.pipelines.sttts_multi import STTTSMultiPipeline as pipeline
    elif config['pipeline'] == "nac":
        shell_run('anonymization/pipelines/nac/install.sh')
        sys.path.append(os.path.join(os.path.dirname(__file__), 'anonymization/modules/nac/coqui_tts/'))
        if devices[0] == torch.device('cpu'):
            from anonymization.pipelines.nac.nac_pipeline import NACPipeline as pipeline
        else:
            from anonymization.pipelines.nac.nac_pipeline_accelerate import NACPipeline as pipeline
    elif config['pipeline'] == "asrbn":
        shell_run('anonymization/pipelines/asrbn/install.sh')
        check_dependencies('anonymization/pipelines/asrbn/requirements.txt')
        from anonymization.pipelines.asrbn import ASRBNPipeline as pipeline
    elif config['pipeline'] == "ssl":
        shell_run('anonymization/pipelines/ssl/install.sh')
        check_dependencies('anonymization/pipelines/ssl/requirements.txt')
        from anonymization.pipelines.ssl.ssl_pipeline import SSLPipeline as pipeline
    elif config['pipeline'] == "template":
        from anonymization.pipelines.template import TemplatePipeline as pipeline
    else:
        raise ValueError(f"Pipeline {config['pipeline']} not defined/imported")

    logger.info(f'Running pipeline: {config["pipeline"]}')
    p = pipeline(config=config, force_compute=args.force_compute, devices=devices)
    p.run_anonymization_pipeline(datasets)
