import abc_atlas_access.abc_atlas_cache.file_attributes as abc_attributes
import abc_atlas_access.abc_atlas_cache.abc_project_cache as abc_cache


class AbcCacheWrapper(abc_cache.AbcProjectCache):
    """
    A cache to allow us to download taxonomy files
    using the abc_atlas_access infrastructure in the interim
    before those files are technically released
    """
    def __init__(self, *args, **kwargs):
        self._patch_manifest = dict()
        super().__init__(*args, **kwargs)

    def get_data_path(
            self,
            directory: str,
            file_name: str,
            force_download: bool = False,
            skip_hash_check: bool = False):

        try:
            result = super().get_data_path(
                directory=directory,
                file_name=file_name,
                force_download=force_download,
                skip_hash_check=skip_hash_check
            )
            return result

        except KeyError:
            self._get_mmc_file_patch(
                directory=directory,
                file_name=file_name
            )

            return self._patch_manifest[directory][file_name]

    def _get_mmc_file_patch(
            self,
            directory: str,
            file_name: str):
        """
        Attempt to download data that is in the S3 bucket but
        not actually released yet. Cache the path of the file
        for later use.
        """
        version_lookup = {
            'SEAAD-taxonomy': '20240831',
            'mmc-gene-mapper': '20250630',
            'WHB-taxonomy': '20240831',
            'HMBA-BG-taxonomy-CCN20250428': '20250630',
            'WMB-taxonomy': '20240831'
        }

        version = version_lookup[directory]
        if file_name.startswith('precomputed'):
            suffix = 'h5'
        elif file_name.startswith('query_markers'):
            suffix = 'json'
        elif file_name.startswith('mmc_gene_mapper'):
            suffix = 'db'
        elif file_name.startswith('mouse_markers'):
            suffix = 'json'
        else:
            raise ValueError(
                f"Unclear suffix for file_name {file_name}"
            )

        relative_path = (
            f"mapmycells/{directory}/{version}/{file_name}.{suffix}"
        )

        (local_path,
         success) = self._download_patched_file(
            relative_path=relative_path,
            version=version)

        if success:
            if directory not in self._patch_manifest:
                self._patch_manifest[directory] = dict()
            self._patch_manifest[directory][file_name] = local_path

    def _download_patched_file(
            self,
            relative_path,
            version):

        url = (
            "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/"
            f"{relative_path}"
        )

        local_path = self.cache._manifest._cache_dir / relative_path

        if local_path.exists():
            success = True
        else:
            success = False
            print(f'attempting to download {url}')
            file_attributes = abc_attributes.CacheFileAttributes(
                url=url,
                version=version,
                file_size=0,
                local_path=local_path,
                relative_path=relative_path,
                file_type='placeholder',
                file_hash='1234'
            )

            try:
                success = self.cache._download_file(
                   file_attributes=file_attributes,
                   force_download=False,
                   skip_hash_check=True
                )
            except Exception:
                success = False
            finally:
                if not success:
                    if local_path.exists():
                        print(f'removing {local_path}')
                        local_path.unlink()

        return local_path, success
