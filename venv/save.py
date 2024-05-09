import numpy as np
import h5py
import os.path


class HdF5Saver:
    def __init__(
            self,
            x_gal,
            n_iter_bub,
            n_inside_tau,
            # x_first_bubble,
            output_dir,
            create=True,
            # x_main = 0.0,
            # y_main = 0.0,
            # z_main = 0.0,
            # r_main = 10.0
    ):
        """Saving is done by referencing the first galaxies x-position and
        the first bubbles positin"""
        self.x_gal = x_gal

        if type(n_iter_bub) is tuple:
            n_iter_bub = n_iter_bub[0]
        if type(n_inside_tau) is tuple:
            n_inside_tau = n_inside_tau[0]

        self.n_iter_bub = n_iter_bub,
        self.n_inside_tau = n_inside_tau,
        # self.x_first_bubble = x_first_bubble
        self.output_dir = output_dir

        # self.x_main = x_main
        # self.y_main = y_main
        # self.z_main = z_main
        # self.r_main = r_main
        if create:
            self.create = False
            if not os.path.isdir(output_dir):
                try:
                    os.mkdir(output_dir)
                except FileExistsError:
                    pass
            self.create_file()
        else:
            self.create = True
            self.open()

    def create_file(self):
        print(str(self.n_iter_bub))
        if hasattr(self.n_inside_tau, '__len__'):
            nit = self.n_inside_tau[0]
        else:
            nit = self.n_inside_tau
        if hasattr(self.n_iter_bub, '__len__'):
            nib = self.n_iter_bub[0]
        else:
            nib = self.n_iter_bub
        self.fname = (self.output_dir +
                      f"{self.x_gal:.8f}" + "_" +
                      str(nib) + "_" +
                      str(nit) +
                      '.hdf5')
        # f"{self.x_first_bubble:.4f}" + "_" +
        # f"{self.x_main:.2f}" + "_" +
        # f"{self.y_main:.2f}" + "_" +
        # f"{self.z_main:.2f}" + "_" +
        # f"{self.r_main:.2f}" + "_" +
        # '.hdf5')
        self.f = h5py.File(self.fname, 'a')

        self.created = True

    def open(self):
        self.fname = (self.output_dir +
                      f"{self.x_gal:.4f}" + "_" +
                      f"{self.n_iter_bub}" + "_" +
                      f"{self.n_inside_tau}" +
                      '.hdf5')
        # f"{self.x_first_bubble:.4f}" + "_" +
        # f"{self.x_main:.2f}" + "_" +
        # f"{self.y_main:.2f}" + "_" +
        # f"{self.z_main:.2f}" + "_" +
        # f"{self.r_main:.2f}" + "_" +
        # '.hdf5')
        self.f = h5py.File(self.fname, 'a')

    def save_attrs(self, dict_gal):
        for (nam, val) in dict_gal.items():
            self.f.attrs[nam] = val

    def save_datasets(self, dict_dat):
        for (nam, val) in dict_dat.items():
            self.f.create_dataset(
                nam,
                dtype="float",
                data=val
            )

    def close_file(self):
        self.f.close()


class HdF5SaverAft:
    def __init__(
            self,
            f_name,
    ):
        self.f = None
        self.f_name = f_name
        self.open()

    def open(self):
        self.f = h5py.File(self.f_name, 'a')

    def save_data_after(self, x_main, y_main, z_main, r_bub_main, dict_dat):
        """
        Function will save data inside the likelihood functon
        :return:
        """
        group_name = str(
            x_main
        ) + "_" + str(
            y_main
        ) + "_" + str(
            z_main
        ) + "_" + str(
            r_bub_main
        )
        f_group = self.f.create_group(
            group_name
        )
        for (nam, val) in dict_dat.items():
            f_group.create_dataset(
                nam,
                dtype="float",
                data=val
            )
    def close(self):
        self.f.close()
