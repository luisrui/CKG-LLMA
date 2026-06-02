#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>

typedef unsigned int ui;

using namespace std;
namespace py = pybind11;

int randint_(int low, int high)
{
    static std::random_device rd; 
    static std::mt19937 gen(rd()); 
    std::uniform_int_distribution<> distrib(low, high); 

    return distrib(gen); 
}

py::array_t<int> sample_negative(py::array_t<int> users, py::array_t<int> items, py::dict allPos, int num_users, int num_items) {
    auto users_buf = users.request();
    auto items_buf = items.request();
    int *users_ptr = (int *)users_buf.ptr;
    int *items_ptr = (int *)items_buf.ptr;
    int batch_size = users_buf.shape[0];
    int neg_num = items_buf.shape[0];
    int row = neg_num + 2;
    
    py::array_t<int> S_array = py::array_t<int>({batch_size, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int idx = 0; idx < batch_size; idx++) {
        int user = users_ptr[idx];
        int positem = items_ptr[idx];

        py::list pos_item_list = allPos[py::int_(user)];
        std::unordered_set<int> pos_item_set;
        for (auto item : pos_item_list) {
            pos_item_set.insert(item.cast<int>());
        }

        ptr[idx * row] = user;
        ptr[idx * row + 1] = positem;

        for (int i = 2; i < neg_num + 2; i++) {
            int negitem;
            do {
                negitem = randint_(num_users, num_users + num_items - 1);
            } while (pos_item_set.find(negitem) != pos_item_set.end());
            ptr[idx * row + i] = negitem;
        }
    }
    return S_array;
}

void set_seed(unsigned int seed)
{
    srand(seed);
}

using namespace py::literals;

PYBIND11_MODULE(SampleFunction, m)
{
    srand(time(0));
    // srand(2020);
    m.doc() = "example plugin";
    m.def("randint", &randint_, "generate int between [0 end]", "low", "high");
    m.def("seed", &set_seed, "set random seed", "seed");
    m.def("sample_negative", &sample_negative, "sampling negatives for all",
          "user", "item", "allPos", "num_users", "num_items");
}
/*
<%
cfg['compiler_args'] = ['-std=c++11', '-undefined dynamic_lookup']
%>
<%
setup_pybind11(cfg)
%>
*/