#ifndef DOUBLELINKEDLIST_H__
#define DOUBLELINKEDLIST_H__

#include <iosfwd>

template <class T>
struct DoubleNode
{
  DoubleNode (DoubleNode<T> *Prev, DoubleNode<T> *Next, T Value) : prev(Prev), next(Next), value(Value) {}

  DoubleNode<T> *prev, *next;
  T value;
};

template <class T>
class DoubleLinkedList
{
 public:
  DoubleLinkedList () :root(&root,&root,T()), initialized(0), current(&root) {}
  
  ~DoubleLinkedList();
  
  void append(T value);
  
  void print();
  
  DoubleNode<T> root;

  // Gets the current value.
  T get();
  // Returns the value of the previous or next position.
  T read_next();
  T read_prev();
  // Get the next or previous value and move current to that pointer.
  T next();
  T prev();

private:
  bool initialized;
  DoubleNode<T> *current;
};

// Always appends to the end of the chain (before the root, which closes the cycle)
template <class T>
void DoubleLinkedList<T>::append(T value)
{
  if (initialized == false)
    {
      root.value = value;
      initialized = true;
    }
  else
    {
      DoubleNode<T> *node = new DoubleNode<T>(root.prev, &root, value);
      root.prev->next = node;
      root.prev = node;
    }
}

template <class T>
DoubleLinkedList<T>::~DoubleLinkedList()
{
  DoubleNode<T> *node = root.next, *next_node;
  while (node != &root)
    {
      next_node = node->next;
      delete node;
      node = next_node;
    }
}

template <class T>
T DoubleLinkedList<T>::get()
{
  return current->value;
}

template <class T>
T DoubleLinkedList<T>::next()
{
  current = current->next;
  return current->value;
}

template <class T>
T DoubleLinkedList<T>::prev()
{
  current = current->prev;
  return current->value;
}

template <class T>
T DoubleLinkedList<T>::read_next()
{
  return current->next->value;
}

template <class T>
T DoubleLinkedList<T>::read_prev()
{
  return current->prev->value;
}


template <class T>
void DoubleLinkedList<T>::print()
{
  DoubleNode<T> *node = root.next;

  std::cout << "Fwd: ";
  std::cout << root.value << " ";
  while (node != &root) // More than one node.
    {
      std::cout << node->value << " ";
      node = node->next;
    }
  std::cout << std::endl;
}


#endif // DOUBLELINKEDLIST_H__



